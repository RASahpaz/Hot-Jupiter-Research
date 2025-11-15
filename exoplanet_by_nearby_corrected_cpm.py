#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 11:58:34 2025
Author: Resul Ayberk Şahpaz (merged & debugged)
Description: Hot Jupiter companion search pipeline with two CPM metrics
"""

from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.gaia import Gaia
import pandas as pd
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from uncertainties import ufloat
from uncertainties.umath import sqrt as usqrt

# ==============================================================
# 0. Ayarlar
# ==============================================================
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # just a reminder

# ==============================================================
# 1. GAIA ID SÜTUNUNU OTOMATİK BUL (güvenilir şekilde)
# ==============================================================
print("🔍 NASA Exoplanet Archive sütunları inceleniyor...")

tbl = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="top 1 *"
)

# Orijinal isimleri alıp lower() ile kontrol et, ama sakla orijinal isim
colnames = list(tbl.colnames)
gaia_col = None
for c in colnames:
    if "gaia" in c.lower() and "id" in c.lower():
        gaia_col = c
        break

if gaia_col:
    print(f"✅ Gaia ID sütunu bulundu: {gaia_col}")
else:
    print("⚠️ Gaia ID sütunu bulunamadı, RA–Dec ile eşleştirme yapılacak.")

# ==============================================================
# 2. SICAK JÜPİTERLERİ ÇEK
# ==============================================================
print("\n🔭 Exoplanet Archive'dan sıcak Jüpiter sistemleri çekiliyor...")

select_fields = ["pl_name", "hostname", "pl_orbper", "pl_bmassj", "ra", "dec"]
if gaia_col:
    select_fields.append(gaia_col)

hot_jupiters = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select=",".join(select_fields),
    where="pl_bmassj > 0.3 AND pl_orbper < 10"
).to_pandas()

hot_jupiters = hot_jupiters.dropna(subset=["ra", "dec"])
print(f"→ {len(hot_jupiters)} adet sıcak Jüpiter bulundu.")

# ==============================================================
# 3. GAIA'DA YAKIN KAYNAKLARI ARA
# ==============================================================
def search_gaia_nearby(ra, dec, radius_arcsec=5):
    """Belirtilen RA/Dec çevresinde Gaia kaynaklarını getirir (pandas DataFrame)."""
    radius_deg = radius_arcsec / 3600.0
    query = f"""
    SELECT source_id, ra, dec, parallax, pmra, pmdec, pmra_error, pmdec_error, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    """
    job = Gaia.launch_job_async(query, dump_to_file=False)
    res = job.get_results()
    return res.to_pandas()

# ==============================================================
# 4. AÇISAL AYRILIK (SkyCoord kullanarak) - consistent name
# ==============================================================
def angular_sep(ra1, dec1, ra2, dec2):
    """Açısal ayrımı (arcsec) hesaplar, SkyCoord ile."""
    c1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
    c2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
    return c1.separation(c2).arcsecond

# ==============================================================
# 5. CPM METRIKLERİ
#    - simple_pm_diff: vektörel proper-motion farkının normu (mas/yr)
#    - mugrauer_cpm: Mugrauer tarzı cpm-index (hata ile birlikte, ufloat)
# ==============================================================
def simple_pm_diff(pmra1, pmdec1, pmra2, pmdec2):
    """Basit PM farkı (mas/yr) — vektör farkın büyüklüğü."""
    if any(pd.isna([pmra1, pmdec1, pmra2, pmdec2])):
        return np.nan
    return np.sqrt((pmra1 - pmra2)**2 + (pmdec1 - pmdec2)**2)

def mugrauer_cpm(pmra1, pmra1_err, pmdec1, pmdec1_err,
                 pmra2, pmra2_err, pmdec2, pmdec2_err):
    """
    Mugrauer et al.-like CPM index:
        pm1 = sqrt(pmra1^2 + pmdec1^2)
        pm2 = sqrt(pmra2^2 + pmdec2^2)
        cpm = (pm1 + pm2) / abs(pm1 - pm2)
    Returns (cpm_nominal, cpm_std) or (None, None) if insufficient data.
    """
    vals = [pmra1, pmra1_err, pmdec1, pmdec1_err, pmra2, pmra2_err, pmdec2, pmdec2_err]
    if any(pd.isna(vals)):
        return None, None

    try:
        p_ra1 = ufloat(pmra1, pmra1_err)
        p_dec1 = ufloat(pmdec1, pmdec1_err)
        p_ra2 = ufloat(pmra2, pmra2_err)
        p_dec2 = ufloat(pmdec2, pmdec2_err)

        pm1 = usqrt(p_ra1**2 + p_dec1**2)
        pm2 = usqrt(p_ra2**2 + p_dec2**2)

        denom = abs(pm1 - pm2)
        # avoid division by zero
        if denom.nominal_value == 0:
            return None, None

        cpm = (pm1 + pm2) / denom
        return cpm.nominal_value, cpm.std_dev
    except Exception:
        return None, None

# ==============================================================
# 6. Güvenli Gaia sorgulama: source_id ile sorgu, None dönerse uyarı ver
# ==============================================================
def query_gaia_source_safe(source_id):
    adql = f"""
    SELECT source_id, ra, dec, parallax, pmra, pmdec, pmra_error, pmdec_error, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE source_id = {int(source_id)}
    """
    job = Gaia.launch_job(adql)
    res = job.get_results()
    if len(res) == 0:
        return None
    return res[0]

# ==============================================================
# 7. pipeline: sıcak Jüpiterler etrafında yakın kaynakları tara, iki CPM metriğini hesapla
# ==============================================================
results = []

for idx, row in hot_jupiters.iterrows():
    hostname = row.get("hostname", "")
    pl_name = row.get("pl_name", "")

    print(f"\n🌟 {hostname} ({pl_name}) inceleniyor...")

    try:
        nearby = search_gaia_nearby(row["ra"], row["dec"], radius_arcsec=5)
    except Exception as e:
        print(f" ❌ Gaia sorgusu başarısız: {e}")
        continue

    if len(nearby) <= 1:
        print(" → Yakın kaynak yok.")
        continue

    nearby_count = len(nearby) - 1  # ana yıldız hariç

    # primary seçimi: eğer exoplanet tablosunda bir Gaia ID varsa (gaia_col), ona göre seç; yoksa en parlakı al
    primary = None
    if gaia_col and gaia_col in row and not pd.isna(row[gaia_col]):
        try:
            # bazen gaia_col bir string tipinde id içerir; güvenli int dönüşümü dene
            primary_srcid = int(row[gaia_col])
            match = nearby.loc[nearby["source_id"] == primary_srcid]
            if len(match) > 0:
                primary = match.iloc[0]
        except Exception:
            primary = None

    if primary is None:
        # en parlak (en küçük G) kaydı ana kabul et
        primary = nearby.sort_values("phot_g_mean_mag").iloc[0]

    # Komşular için CPM hesabı
    for _, cand in nearby.iterrows():
        if int(cand["source_id"]) == int(primary["source_id"]):
            continue

        # açısal ayrılık
        sep_arcsec = angular_sep(primary["ra"], primary["dec"], cand["ra"], cand["dec"])

        # simple pm diff
        pm_diff = simple_pm_diff(primary.get("pmra"), primary.get("pmdec"),
                                 cand.get("pmra"), cand.get("pmdec"))

        # mugrauer style cpm (with errors)
        cpm_val, cpm_err = mugrauer_cpm(
            primary.get("pmra"), primary.get("pmra_error"),
            primary.get("pmdec"), primary.get("pmdec_error"),
            cand.get("pmra"), cand.get("pmra_error"),
            cand.get("pmdec"), cand.get("pmdec_error"),
        )

        # seçim kriteri: örnek olarak simple pm_diff < 5 mas/yr ekliyoruz
        # ayrıca Mugrauer cpm varsa onu da kaydet
        if np.isfinite(pm_diff) and pm_diff < 5:
            results.append({
                "host": hostname,
                "planet": pl_name,
                "primary_gaia": int(primary["source_id"]),
                "candidate_gaia": int(cand["source_id"]),
                "sep_arcsec": sep_arcsec,
                "pm_diff_masyr": pm_diff,
                "mugrauer_cpm": cpm_val,
                "mugrauer_cpm_err": cpm_err,
                "nearby_count": nearby_count
            })

    print(f" → {nearby_count} komşu bulundu, olası CPM eşleşmeleri eklendi.")

# ==============================================================
# 8. SONUÇLAR ve KAYIT
# ==============================================================
df = pd.DataFrame(results)
if not df.empty:
    df = df.sort_values(by=["mugrauer_cpm", "pm_diff_masyr"], na_position="last")
print("\n✅ Analiz tamamlandı!")
print(f"Toplam olası bağlı aday sayısı: {len(df)}")
print(df.head(10))

output_file = "hot_jupiter_cpm_candidates.csv"
df.to_csv(output_file, index=False)
print(f"💾 Olası bağlı adaylar ve yakın cisim sayıları '{output_file}' dosyasına kaydedildi.")
