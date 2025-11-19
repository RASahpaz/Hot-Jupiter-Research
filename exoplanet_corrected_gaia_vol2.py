#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 11:58:34 2025
Author: Resul Ayberk Şahpaz (merged & debugged)
Description: Hot Jupiter companion search pipeline with robust Gaia ID resolution
"""

from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.gaia import Gaia
import pandas as pd
import numpy as np
import re
from astropy import units as u
from astropy.coordinates import SkyCoord
from uncertainties import ufloat
from uncertainties.umath import sqrt as usqrt

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

# ==============================================================
# 1. GAIA ID SÜTUNUNU OTOMATİK BUL
# ==============================================================
print("🔍 NASA Exoplanet Archive sütunları inceleniyor...")

tbl = NasaExoplanetArchive.query_criteria(table="pscomppars", select="top 1 *")
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
# Helper Functions
# ==============================================================

def angular_sep(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    return np.degrees(2 * np.arcsin(np.sqrt(np.sin((dec1-dec2)/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin((ra1-ra2)/2)**2))) * 3600


def simple_pm_diff(pmra1, pmdec1, pmra2, pmdec2):
    if any(pd.isna([pmra1, pmdec1, pmra2, pmdec2])):
        return np.nan
    return np.sqrt((pmra1 - pmra2)**2 + (pmdec1 - pmdec2)**2)


def mugrauer_cpm(pmra1, pmra1_err, pmdec1, pmdec1_err, pmra2, pmra2_err, pmdec2, pmdec2_err):
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
    except Exception:
        return None, None


def search_gaia_nearby(ra, dec, radius_arcsec=5):
    radius_deg = radius_arcsec / 3600.0
    query = f"""
    SELECT source_id, ra, dec, parallax, pmra, pmdec, pmra_error, pmdec_error, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
    """
    job = Gaia.launch_job_async(query, dump_to_file=False)
    res = job.get_results()
    return res.to_pandas()


def resolve_primary(row, nearby, gaia_col=None, max_sep_arcsec=3.0, mag_tol=2.0):
    nearby = nearby.copy()
    if 'source_id' in nearby.columns:
        try:
            nearby['source_id'] = nearby['source_id'].astype('int64')
        except Exception:
            pass

    if gaia_col and gaia_col in row and not pd.isna(row[gaia_col]):
        raw = str(row[gaia_col]).strip()
        try:
            cid = int(float(raw))
            m = nearby.loc[nearby['source_id'] == cid]
            if len(m) > 0:
                return m.iloc[0]
        except Exception:
            pass
        m = re.search(r'(\d{9,20})', raw)
        if m:
            try:
                cid = int(m.group(1))
                m2 = nearby.loc[nearby['source_id'] == cid]
                if len(m2) > 0:
                    return m2.iloc[0]
            except Exception:
                pass

    nearby['sep_arcsec'] = nearby.apply(lambda r: angular_sep(row['ra'], row['dec'], r['ra'], r['dec']), axis=1)
    nearby_sorted = nearby.sort_values('sep_arcsec')
    if len(nearby_sorted) == 0:
        return None

    best = nearby_sorted.iloc[0]
    if best['sep_arcsec'] > max_sep_arcsec:
        print(f"⚠️ {row.get('hostname','?')} için en yakın Gaia kaynağı {best['sep_arcsec']:.2f}" "(limit {max_sep_arcsec}).")

    if 'phot_g_mean_mag' in row and not pd.isna(row['phot_g_mean_mag']) and not pd.isna(best.get('phot_g_mean_mag', np.nan)):
        try:
            diff = abs(best['phot_g_mean_mag'] - float(row['phot_g_mean_mag']))
            if diff > mag_tol:
                print(f"⚠️ Mag mismatch for {row.get('hostname','?')}: candidate G={best['phot_g_mean_mag']} vs catalog {row['phot_g_mean_mag']} Δ={diff:.2f} mag")
        except Exception:
            pass
    return best

# ==============================================================
# 3. Pipeline
# ==============================================================
results = []
suspects = []

for idx, row in hot_jupiters.iterrows():
    hostname = row.get('hostname', '')
    pl_name = row.get('pl_name', '')
    try:
        nearby = search_gaia_nearby(row['ra'], row['dec'], radius_arcsec=5)
    except Exception as e:
        print(f"❌ Gaia sorgusu başarısız: {e}")
        continue

    if len(nearby) <= 1:
        print(f"→ {hostname}: Yakın kaynak yok.")
        continue

    primary = resolve_primary(row, nearby, gaia_col=gaia_col)
    if primary is None:
        print(f"❌ {hostname}: Primary bulunamadı.")
        continue

    nearby_count = len(nearby) - 1

    for _, cand in nearby.iterrows():
        if int(cand['source_id']) == int(primary['source_id']):
            continue
        sep_arcsec = angular_sep(primary['ra'], primary['dec'], cand['ra'], cand['dec'])
        pm_diff = simple_pm_diff(primary.get('pmra'), primary.get('pmdec'), cand.get('pmra'), cand.get('pmdec'))
        cpm_val, cpm_err = mugrauer_cpm(primary.get('pmra'), primary.get('pmra_error'), primary.get('pmdec'), primary.get('pmdec_error'), cand.get('pmra'), cand.get('pmra_error'), cand.get('pmdec'), cand.get('pmdec_error'))
        if np.isfinite(pm_diff) and pm_diff < 5:
            results.append({
                'host': hostname,
                'planet': pl_name,
                'primary_gaia': int(primary['source_id']),
                'candidate_gaia': int(cand['source_id']),
                'sep_arcsec': sep_arcsec,
                'pm_diff_masyr': pm_diff,
                'mugrauer_cpm': cpm_val,
                'mugrauer_cpm_err': cpm_err,
                'nearby_count': nearby_count
            })
        if sep_arcsec > 3 or ('phot_g_mean_mag' in row and not pd.isna(row.get('phot_g_mean_mag', np.nan)) and not pd.isna(cand.get('phot_g_mean_mag', np.nan)) and abs(cand['phot_g_mean_mag'] - row['phot_g_mean_mag']) > 2):
            suspects.append({
                'host': hostname,
                'planet': pl_name,
                'raw_gaia_col': row.get(gaia_col, ''),
                'chosen_source_id': int(cand['source_id']),
                'sep_arcsec': sep_arcsec,
                'phot_g_catalog': row.get('phot_g_mean_mag', np.nan),
                'phot_g_candidate': cand.get('phot_g_mean_mag', np.nan)
            })

# ==============================================================
# 4. Save results
# ==============================================================
df = pd.DataFrame(results)
if not df.empty:
    df = df.sort_values(by=['mugrauer_cpm', 'pm_diff_masyr'], na_position='last')
df.to_csv('hot_jupiter_cpm_candidates_corrected.csv', index=False)
print(f"💾 Olası bağlı adaylar '{'hot_jupiter_cpm_candidates_corrected.csv'}' dosyasına kaydedildi.")

if len(suspects) > 0:
    pd.DataFrame(suspects).to_csv('suspicious_matches.csv', index=False)
    print("🔎 Şüpheli eşleşmeler 'suspicious_matches.csv' dosyasına kaydedildi.")

print("✅ Analiz tamamlandı!")
