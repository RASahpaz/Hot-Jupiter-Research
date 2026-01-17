#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 13:32:36 2026

@author: resulayberksahpaz
"""

import sys
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.gaia import Gaia
import pandas as pd
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from uncertainties import ufloat
from uncertainties.umath import sqrt as usqrt

# Gaia Tablo Ayarı
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

# ============================================================
# 1. YARDIMCI FONKSİYONLAR
# ============================================================

def angular_sep(ra1, dec1, ra2, dec2):
    try:
        c1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
        c2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
        return c1.separation(c2).arcsecond
    except:
        return np.nan

def search_gaia_nearby(ra, dec, radius_arcsec=15):
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
    try:
        job = Gaia.launch_job_async(query, dump_to_file=False)
        return job.get_results().to_pandas()
    except:
        return pd.DataFrame()

def mugrauer_cpm(pmra1, pmra1_err, pmdec1, pmdec1_err, pmra2, pmra2_err, pmdec2, pmdec2_err):
    vals = [pmra1, pmra1_err, pmdec1, pmdec1_err, pmra2, pmra2_err, pmdec2, pmdec2_err]
    if any(pd.isna(vals)): return np.nan
    try:
        p1 = usqrt(ufloat(pmra1,pmra1_err)**2 + ufloat(pmdec1,pmdec1_err)**2)
        p2 = usqrt(ufloat(pmra2,pmra2_err)**2 + ufloat(pmdec2,pmdec2_err)**2)
        denom = abs(p1 - p2)
        if denom.nominal_value == 0: return np.nan
        cpm = (p1+p2)/denom
        return cpm.nominal_value
    except:
        return np.nan

# ============================================================
# 2. NASA VERİLERİ (680+ Gezegen)
# ============================================================
print("🚀 Pipeline başlatılıyor...")

select_columns = """
    pl_name, hostname, pl_orbper, pl_bmassj,
    ra, dec, sy_dist, gaia_dr3_id,
    sy_pmra, sy_pmdec, sy_plx,
    sy_pmraerr1, sy_pmraerr2, sy_pmdecerr1, sy_pmdecerr2
"""

try:
    master_df = NasaExoplanetArchive.query_criteria(
        table="pscomppars",
        select=select_columns,
        where="pl_bmassj > 0.3 AND pl_orbper < 10"
    ).to_pandas()
except Exception as e:
    print(f"❌ NASA Sorgu Hatası: {e}")
    sys.exit(1)

# Standartlaştırma
master_df = master_df.rename(columns={
    "sy_gaia_id": "gaia_dr3_id",
    "sy_pmra": "host_pmra",
    "sy_pmdec": "host_pmdec",
    "sy_plx": "host_parallax"
})
master_df["host_pmra_error"] = (master_df["sy_pmraerr1"].abs() + master_df["sy_pmraerr2"].abs()) / 2
master_df["host_pmdec_error"] = (master_df["sy_pmdecerr1"].abs() + master_df["sy_pmdecerr2"].abs()) / 2
master_df["gaia_dr3_id"] = master_df["gaia_dr3_id"].astype(str).str.replace("Gaia DR3 ", "", regex=False)

print(f"✅ NASA Arşivi Çekildi: {len(master_df)} gezegen.")

# ============================================================
# 3. GAIA CPM TARAMASI
# ============================================================
unique_hosts = master_df.drop_duplicates(subset=["hostname"])
cpm_results = []

print("🌌 Gaia yoldaş taraması yapılıyor...")
for idx, row in unique_hosts.iterrows():
    nearby = search_gaia_nearby(row["ra"], row["dec"])
    if len(nearby) <= 1: continue
    
    # Host tespiti
    host_gaia = nearby.sort_values("phot_g_mean_mag").iloc[0] # Basitçe en parlak olan
    
    for _, cand in nearby.iterrows():
        if cand["source_id"] == host_gaia["source_id"]: continue
        if pd.isna(cand["pmra"]): continue
        
        pm_diff = np.sqrt((host_gaia["pmra"] - cand["pmra"])**2 + (host_gaia["pmdec"] - cand["pmdec"])**2)
        if pm_diff < 15: # Eşik
            cpm_score = mugrauer_cpm(host_gaia["pmra"], host_gaia["pmra_error"], host_gaia["pmdec"], host_gaia["pmdec_error"],
                                     cand["pmra"], cand["pmra_error"], cand["pmdec"], cand["pmdec_error"])
            
            cpm_results.append({
                "hostname": row["hostname"],
                "companion_gaia_id": cand["source_id"],
                "companion_mag": cand["phot_g_mean_mag"],
                "cpm_score": cpm_score,
                "sep_arcsec": angular_sep(host_gaia["ra"], host_gaia["dec"], cand["ra"], cand["dec"])
            })
            break # İlk bulunanı al

cpm_df = pd.DataFrame(cpm_results)

# ============================================================
# 4. BİRLEŞTİRME (NASA + GAIA + DIŞ DOSYA)
# ============================================================

# Önce Gaia sonuçlarını NASA verilerine ekle
merged_df = pd.merge(master_df, cpm_df, on="hostname", how="left")

# DIŞ DOSYAYI OKU (Separator düzeltildi)
try:
    # sep=';' ekledik çünkü dosyan noktalı virgül kullanıyor
    ext_df = pd.read_csv("companion_checked_hot_jupiters.csv", sep=';')
    
    # Dosyadaki sütun isimleri: 'Planet Name', 'Companion Status' vb.
    # NASA verisindeki 'pl_name' ile eşleştiriyoruz.
    final_df = pd.merge(merged_df, ext_df, left_on="pl_name", right_on="Planet Name", how="left")
    print("✅ Dış dosya (noktalı virgül ile) başarıyla birleştirildi.")
except Exception as e:
    print(f"⚠️ Dış dosya okunurken hata: {e}")
    final_df = merged_df

# Gereksiz sütunları temizle ve kaydet
final_df.to_csv("FINAL_MERGED_DATA.csv", index=False)
print(f"🎉 İşlem bitti! Toplam {len(final_df)} satır 'FINAL_MERGED_DATA.csv' dosyasına yazıldı.")