#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 11:58:34 2025
Author: Resul Ayberk Şahpaz
Description: Hot Jupiter companion search pipeline with CPM metrics + Parallax Distance & Agreement
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
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

# ==============================================================
# Parallax fonksiyonları
# ==============================================================
def parallax_distance(p, p_err):
    """Parallax → distance (pc), ufloat ile (p mas)."""
    if pd.isna(p) or p <= 0:
        return np.nan
    return (ufloat(p/1000.0, p_err/1000.0))**-1

def parallax_difference(p1, p1_err, p2, p2_err):
    try:
        u1 = ufloat(p1, p1_err)
        u2 = ufloat(p2, p2_err)
        d = u1 - u2
        return d.nominal_value, d.std_dev
    except:
        return np.nan, np.nan

def parallax_agreement(p1, p1_err, p2, p2_err):
    vals = [p1, p1_err, p2, p2_err]
    if any(pd.isna(vals)):
        return np.nan
    return abs(p1 - p2) / np.sqrt(p1_err**2 + p2_err**2)

# ==============================================================
# 1. GAIA ID sütunu tespiti
# ==============================================================
print("🔍 NASA Exoplanet Archive sütunları inceleniyor...")

tbl = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="top 1 *"
)

colnames = list(tbl.colnames)
gaia_col = None
for c in colnames:
    if c == "gaia_dr3_id":
        gaia_col = c
        break

if gaia_col:
    print(f"✅ Gaia ID sütunu bulundu: {gaia_col}")
else:
    print("⚠️ Gaia ID sütunu bulunamadı, RA/Dec ile eşleştirme yapılacak.")

# ==============================================================
# 2. Sıcak Jüpiterleri çek
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
# 3. Gaia yakın kaynak araması
# ==============================================================
def search_gaia_nearby(ra, dec, radius_arcsec=5):
    radius_deg = radius_arcsec / 3600.0
    query = f"""
    SELECT source_id, ra, dec, parallax, parallax_error,
           pmra, pmra_error, pmdec, pmdec_error, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    """
    job = Gaia.launch_job_async(query, dump_to_file=False)
    return job.get_results().to_pandas()

# ==============================================================
# 4. Açısal ayrılık
# ==============================================================
def angular_sep(ra1, dec1, ra2, dec2):
    c1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
    c2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
    return c1.separation(c2).arcsecond

# ==============================================================
# 5. CPM metrikleri
# ==============================================================
def simple_pm_diff(pmra1, pmdec1, pmra2, pmdec2):
    if any(pd.isna([pmra1, pmdec1, pmra2, pmdec2])):
        return np.nan
    return np.sqrt((pmra1 - pmra2)**2 + (pmdec1 - pmdec2)**2)

def mugrauer_cpm(pmra1, pmra1_err, pmdec1, pmdec1_err,
                 pmra2, pmra2_err, pmdec2, pmdec2_err):

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
        if denom.nominal_value == 0:
            return None, None

        cpm = (pm1 + pm2) / denom
        return cpm.nominal_value, cpm.std_dev

    except:
        return None, None

# ==============================================================
# 7. Ana pipeline döngüsü
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

    nearby_count = len(nearby) - 1

    # primary seçimi
    primary = None
    if gaia_col and gaia_col in row and not pd.isna(row[gaia_col]):
        try:
            gid = int(row[gaia_col])
            match = nearby.loc[nearby["source_id"] == gid]
            if len(match) > 0:
                primary = match.iloc[0]
        except:
            primary = None

    if primary is None:
        primary = nearby.sort_values("phot_g_mean_mag").iloc[0]

    # NASA tablosundan gelen GAIA DR3 ID (sadece primary için)
    primary_dr3_id = row.get(gaia_col, np.nan)

    # ----------------------------------------------------------
    #   Candidate loop + PARALLAX ENTEGRASYONU
    # ----------------------------------------------------------
    for _, cand in nearby.iterrows():
        if int(cand["source_id"]) == int(primary["source_id"]):
            continue

        sep_arcsec = angular_sep(primary["ra"], primary["dec"], cand["ra"], cand["dec"])

        pm_diff = simple_pm_diff(primary["pmra"], primary["pmdec"],
                                 cand["pmra"], cand["pmdec"])

        cpm_val, cpm_err = mugrauer_cpm(
            primary["pmra"], primary["pmra_error"],
            primary["pmdec"], primary["pmdec_error"],
            cand["pmra"], cand["pmra_error"],
            cand["pmdec"], cand["pmdec_error"],
        )

        # PARALLAX
        dist_primary = parallax_distance(primary["parallax"], primary["parallax_error"])
        dist_cand = parallax_distance(cand["parallax"], cand["parallax_error"])

        par_diff, par_diff_err = parallax_difference(
            primary["parallax"], primary["parallax_error"],
            cand["parallax"], cand["parallax_error"]
        )

        par_sigma = parallax_agreement(
            primary["parallax"], primary["parallax_error"],
            cand["parallax"], cand["parallax_error"]
        )

        # ekleme kriteri
        if np.isfinite(pm_diff) and pm_diff < 5:
            results.append({
                "host": hostname,
                "planet": pl_name,

                "primary_gaia": int(primary["source_id"]),
                "candidate_gaia": int(cand["source_id"]),

                "primary_gaia_dr3_id": primary_dr3_id,
                "candidate_gaia_dr3_id": np.nan,

                "sep_arcsec": sep_arcsec,
                "pm_diff_masyr": pm_diff,
                "mugrauer_cpm": cpm_val,
                "mugrauer_cpm_err": cpm_err,
                "dist_primary_pc": dist_primary,
                "dist_candidate_pc": dist_cand,
                "parallax_diff_mas": par_diff,
                "parallax_diff_err": par_diff_err,
                "parallax_sigma_agreement": par_sigma,
                "nearby_count": nearby_count
            })

    print(f" → {nearby_count} komşu bulundu, olası CPM-parallax eşleşmeleri eklendi.")

# ==============================================================
# 8. Sonuç
# ==============================================================
df = pd.DataFrame(results)
if not df.empty:
    df = df.sort_values(by=["parallax_sigma_agreement", "pm_diff_masyr"], na_position="last")

print("\n✅ Analiz tamamlandı!")
print(f"Toplam olası eşleşme: {len(df)}")
print(df.head(10))

output_file = "hot_jupiter_cpm_parallax_candidates.csv"
df.to_csv(output_file, index=False)
print(f"💾 Sonuçlar '{output_file}' dosyasına kaydedildi.")

