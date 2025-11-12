#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 11:58:34 2025

@author: resulayberksahpaz
"""

# --- HOT JUPITER COMPANION SEARCH PIPELINE (SELF-CHECK VERSION) ---
# Author: Resul Ayberk Şahpaz
# Date: November 2025

from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.gaia import Gaia
import pandas as pd
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

# ==============================================================
# 1. GAIA ID SÜTUNUNU OTOMATİK BUL
# ==============================================================

print("🔍 NASA Exoplanet Archive sütunları inceleniyor...")

tbl = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="top 1 *"
)
columns = [c.lower() for c in tbl.colnames]

# olası isimleri kontrol et
possible_gaia_cols = [c for c in columns if "gaia" in c and "id" in c]
if possible_gaia_cols:
    gaia_col = possible_gaia_cols[0]
    print(f"✅ Gaia ID sütunu bulundu: {gaia_col}")
else:
    gaia_col = None
    print("⚠️ Gaia ID sütunu bulunamadı, RA–Dec ile eşleştirme yapılacak.")

# ==============================================================
# 2. SICAK JÜPİTERLERİ ÇEK
# ==============================================================

print("\n🔭 Exoplanet Archive'dan sıcak Jüpiter sistemleri çekiliyor...")

select_fields = "pl_name,hostname,pl_orbper,pl_bmassj,ra,dec"
if gaia_col:
    select_fields += f",{gaia_col}"

hot_jupiters = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select=select_fields,
    where="pl_bmassj > 0.3 AND pl_orbper > 10"
).to_pandas()

hot_jupiters = hot_jupiters.dropna(subset=["ra", "dec"])
print(f"→ {len(hot_jupiters)} adet sıcak Jüpiter bulundu.")

# ==============================================================
# 3. GAIA'DA YAKIN KAYNAKLARI ARA
# ==============================================================

def search_gaia_nearby(ra, dec, radius_arcsec=5):
    """Belirtilen RA/Dec çevresinde Gaia kaynaklarını getirir."""
    radius_deg = radius_arcsec / 3600
    query = f"""
    SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    """
    job = Gaia.launch_job_async(query, dump_to_file=False)
    return job.get_results().to_pandas()

# ==============================================================
# 4. CPM İNDEKS HESABI
# ==============================================================

def cpm_index(primary, candidate):
    """Proper motion farkını hesaplar (mas/yr)."""
    if np.any(pd.isna([primary['pmra'], primary['pmdec'], candidate['pmra'], candidate['pmdec']])):
        return np.nan
    dmu_ra = primary['pmra'] - candidate['pmra']
    dmu_dec = primary['pmdec'] - candidate['pmdec']
    return np.sqrt(dmu_ra**2 + dmu_dec**2)

def angular_sep(ra1, dec1, ra2, dec2):
    """Açısal ayrımı (arcsec) hesaplar."""
    c1 = SkyCoord(ra1*u.deg, dec1*u.deg)
    c2 = SkyCoord(ra2*u.deg, dec2*u.deg)
    return c1.separation(c2).arcsecond

# ==============================================================
# 5. ANA DÖNGÜ
# ==============================================================

results = []

for idx, row in hot_jupiters.iterrows():
    print(f"\n🌟 {row['hostname']} ({row['pl_name']}) inceleniyor...")

    try:
        nearby = search_gaia_nearby(row['ra'], row['dec'])
    except Exception as e:
        print(f" ❌ Gaia sorgusu başarısız: {e}")
        continue

    if len(nearby) <= 1:
        print(" → Yakın kaynak yok.")
        continue

    nearby_count = len(nearby) - 1  # ana yıldız hariç yakın kaynak sayısı

    # Gaia ID varsa o kaydı al, yoksa en parlak olanı merkez kabul et
    primary = pd.DataFrame()
    if gaia_col and gaia_col in row and not pd.isna(row[gaia_col]):
        try:
            primary = nearby.loc[nearby['source_id'] == int(row[gaia_col])]
        except Exception:
            pass
    if primary.empty:
        primary = nearby.sort_values("phot_g_mean_mag").iloc[[0]]
    primary = primary.iloc[0]

    # Komşular için CPM hesabı
    for _, cand in nearby.iterrows():
        if cand['source_id'] == primary['source_id']:
            continue

        sep = angular_sep(primary['ra'], primary['dec'], cand['ra'], cand['dec'])
        cpm = cpm_index(primary, cand)

        if np.isfinite(cpm) and cpm < 5:
            results.append({
                'host': row['hostname'],
                'planet': row['pl_name'],
                'primary_gaia': int(primary['source_id']),
                'candidate_gaia': int(cand['source_id']),
                'sep_arcsec': sep,
                'cpm_index': cpm,
                'nearby_count': nearby_count
            })

    print(f" → {nearby_count} komşu bulundu, olası CPM eşleşmeleri eklendi.")

# ==============================================================
# 6. SONUÇLAR
# ==============================================================

df = pd.DataFrame(results)
df = df.sort_values('cpm_index')
print("\n✅ Analiz tamamlandı!")
print(f"Toplam olası bağlı aday sayısı: {len(df)}")

print(df.head(10))

# ==============================================================
# 7. SONUÇLARI DOSYAYA KAYDET
# ==============================================================

output_file = "hot_jupiter_cpm_candidates.csv"
df.to_csv(output_file, index=False)
print(f"💾 Olası bağlı adaylar ve yakın cisim sayıları '{output_file}' dosyasına kaydedildi.")
