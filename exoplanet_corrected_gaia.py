#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 11:58:34 2025
Author: Resul Ayberk Şahpaz (merged & debugged)
Description: Hot Jupiter companion search pipeline with improved primary resolver + CPM metrics
"""

import re
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.gaia import Gaia
import pandas as pd
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from uncertainties import ufloat
from uncertainties.umath import sqrt as usqrt

# ==============================================================
# Gaia ayarları
# ==============================================================
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

# ==============================================================
# Güvenli açısal ayrım fonksiyonu
# ==============================================================
def angular_sep(ra1, dec1, ra2, dec2):
    c1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
    c2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
    return c1.separation(c2).arcsecond

# ==============================================================
# YENİ: Primary Resolver (tüm düzeltmeler eklenmiş)
# ==============================================================
def resolve_primary(row, nearby, gaia_col=None, max_sep_arcsec=3.0, mag_tol=2.0):
    """
    row: hot_jupiter satırı
    nearby: Gaia candidate DataFrame
    gaia_col: Exoplanet Archive'ın verdiği Gaia ID kolonu
    """
    # source_id integer olsun
    if 'source_id' in nearby.columns:
        try:
            nearby['source_id'] = nearby['source_id'].astype('int64')
        except Exception:
            pass

    # 1) Eğer NASA tablosunda Gaia ID bulunuyorsa robust parse et
    if gaia_col and gaia_col in row and not pd.isna(row[gaia_col]):
        raw = str(row[gaia_col]).strip()

        # önce float->int (1.23e17 formatı için)
        try:
            cand_id = int(float(raw))
            m = nearby.loc[nearby['source_id'] == cand_id]
            if len(m) > 0:
                return m.iloc[0]
        except Exception:
            pass

        # regex ile uzun ID çek
        m = re.search(r'(\d{9,20})', raw)
        if m:
            try:
                cand_id = int(m.group(1))
                m2 = nearby.loc[nearby['source_id'] == cand_id]
                if len(m2) > 0:
                    return m2.iloc[0]
            except Exception:
                pass
        # koordinat fallback'e düş

    # 2) Fallback: koordinata en yakın kaynağı seç
    def sep_to_row(r):
        return angular_sep(row['ra'], row['dec'], r['ra'], r['dec'])

    nearby = nearby.copy()
    nearby["sep_arcsec"] = nearby.apply(lambda r: sep_to_row(r), axis=1)
    nearby_sorted = nearby.sort_values("sep_arcsec")

    if len(nearby_sorted) == 0:
        return None

    best = nearby_sorted.iloc[0]

    # tolerans dışı ise warning
    if best['sep_arcsec'] > max_sep_arcsec:
        print(f"⚠️ Uyarı: {row.get('hostname','?')} için en yakın Gaia kaynağı {best['sep_arcsec']:.2f}\" uzakta.")

    # magnitude kontrolü varsa (çoğu exoplanet kaydında yok)
    if 'phot_g_mean_mag' in row and not pd.isna(row['phot_g_mean_mag']):
        cand_mag = best.get('phot_g_mean_mag', np.nan)
        if not pd.isna(cand_mag):
            try:
                mag_diff = abs(cand_mag - float(row['phot_g_mean_mag']))
                if mag_diff > mag_tol:
                    print(f"⚠️ Mag mismatch: {row.get('hostname','?')} G={cand_mag}, Catalog={row['phot_g_mean_mag']} (Δ={mag_diff:.2f})")
            except Exception:
                pass

    return best

# ==============================================================
# Gaia Nearby Search
# ==============================================================
def search_gaia_nearby(ra, dec, radius_arcsec=5):
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
    return job.get_results().to_pandas()

# ==============================================================
# CPM METRİKLERİ
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
        p1ra = ufloat(pmra1, pmra1_err)
        p1de = ufloat(pmdec1, pmdec1_err)
        p2ra = ufloat(pmra2, pmra2_err)
        p2de = ufloat(pmdec2, pmdec2_err)

        pm1 = usqrt(p1ra**2 + p1de**2)
        pm2 = usqrt(p2ra**2 + p2de**2)

        denom = abs(pm1 - pm2)
        if denom.nominal_value == 0:
            return None, None

        cpm = (pm1 + pm2) / denom
        return cpm.nominal_value, cpm.std_dev
    except Exception:
        return None, None

# ==============================================================
# NASA EXOPLANET ARCHIVE: GAIA ID kolonu tespiti
# ==============================================================
tbl = NasaExoplanetArchive.query_criteria(
    table="pscomppars", select="top 1 *"
)
cols = list(tbl.colnames)
gaia_col = None
for c in cols:
    if "gaia" in c.lower() and "id" in c.lower():
        gaia_col = c
        break

print(f"Gaia ID sütunu: {gaia_col}")

# ==============================================================
# Sıcak Jüpiterleri Çek
# ==============================================================
select_fields = ["pl_name", "hostname", "pl_orbper", "pl_bmassj", "ra", "dec"]
if gaia_col:
    select_fields.append(gaia_col)

hot_jupiters = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select=",".join(select_fields),
    where="pl_bmassj > 0.3 AND pl_orbper < 10"
).to_pandas()

hot_jupiters = hot_jupiters.dropna(subset=["ra", "dec"])

# ==============================================================
# PIPELINE
# ==============================================================
results = []
suspects = []

for idx, row in hot_jupiters.iterrows():
    hostname = row.get("hostname", "")
    pl_name = row.get("pl_name", "")

    print(f"\n🌟 {hostname} ({pl_name}) inceleniyor...")

    try:
        nearby = search_gaia_nearby(row["ra"], row["dec"], radius_arcsec=5)
    except Exception as e:
        print(" ❌ Gaia sorgusu hata verdi:", e)
        continue

    if len(nearby) <= 1:
        print(" → Komşu yok.")
        continue

    # primary’yi yeni fonksiyonla seç
    primary = resolve_primary(row, nearby, gaia_col=gaia_col)

    if primary is None:
        print(" ❌ Primary bulunamadı, geçiliyor.")
        suspects.append({
            "host": hostname,
            "planet": pl_name,
            "reason": "Primary not found",
            "raw_gaia_col": row.get(gaia_col, "")
        })
        continue

    primary_id = int(primary["source_id"])

    # komşu sayısı (ana dahil değil)
    nearby_count = len(nearby) - 1

    # diğer kaynakları tara
    for _, cand in nearby.iterrows():
        if int(cand["source_id"]) == primary_id:
            continue

        sep_arcsec = angular_sep(primary["ra"], primary["dec"], cand["ra"], cand["dec"])
        pm_diff = simple_pm_diff(primary["pmra"], primary["pmdec"], cand["pmra"], cand["pmdec"])
        cpm_val, cpm_err = mugrauer_cpm(
            primary["pmra"], primary["pmra_error"],
            primary["pmdec"], primary["pmdec_error"],
            cand["pmra"], cand["pmra_error"],
            cand["pmdec"], cand["pmdec_error"]
        )

        if np.isfinite(pm_diff) and pm_diff < 5:
            results.append({
                "host": hostname,
                "planet": pl_name,
                "primary_gaia": primary_id,
                "candidate_gaia": int(cand["source_id"]),
                "sep_arcsec": sep_arcsec,
                "pm_diff_masyr": pm_diff,
                "mugrauer_cpm": cpm_val,
                "mugrauer_cpm_err": cpm_err,
                "nearby_count": nearby_count
            })

# ==============================================================
# Sonuçları kaydet
# ==============================================================
df = pd.DataFrame(results)
if not df.empty:
    df = df.sort_values(by=["mugrauer_cpm", "pm_diff_masyr"], na_position="last")

df.to_csv("hot_jupiter_cpm_candidates_corrected.csv", index=False)
pd.DataFrame(suspects).to_csv("suspicious_matches.csv", index=False)

print("\n✅ Analiz tamamlandı!")
print(f"Toplam aday: {len(df)}")
print("💾 Kayıtlar:")
print(" - hot_jupiter_cpm_candidates_corrected_gaia.csv")
print(" - suspicious_matches.csv")

