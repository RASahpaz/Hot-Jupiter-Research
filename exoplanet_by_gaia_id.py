#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 23:34:11 2025

@author: resulayberksahpaz
"""

import pandas as pd
import numpy as np
import csv
import time
from astropy.table import Table
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.gaia import Gaia
from astropy import units as u
from uncertainties import ufloat, umath
import math

# --- AYARLAR ---
OUTPUT_FILE = "sicak_jupiter_cpm_sonuclari.csv" # Sonuçların kaydedileceği dosya
NASA_TABLE = "PSCompPars"  # En güncel ve tekil parametreler tablosu
SEARCH_RADIUS_ARCSEC = 5.0 # Arama yarıçapı
CPM_THRESHOLD_GOOD = 3.0
CPM_THRESHOLD_GREAT = 10.0

# --- FONKSİYONLAR ---

def get_all_hot_jupiter_hosts():
    """
    Sıcak Jüpiter kriterlerine uyan TÜM yıldızları çeker.
    """
    print(f"📥 VERİ ÇEKİLİYOR: NASA Exoplanet Archive ({NASA_TABLE})...")
    
    # Sütunlar
    select_cols = "hostname, pl_bmassj, pl_orbper, gaia_dr3_id"
    
    # Kriterler (M > 0.3 Mjup, P > 10 gün)
    criteria = "pl_bmassj > 0.3 AND pl_orbper > 10.0 AND gaia_dr3_id IS NOT NULL"
    
    try:
        table = NasaExoplanetArchive.query_criteria(
            table=NASA_TABLE,
            where=criteria,
            select=select_cols,
            cache=False
        )
    except Exception as e:
        print(f"❌ HATA: NASA Arşivine bağlanılamadı. Hata: {e}")
        return []

    df = table.to_pandas()
    
    # Gaia ID temizleme ve benzersizleştirme
    if 'gaia_dr3_id' not in df.columns:
        print("❌ HATA: 'gaia_dr3_id' sütunu bulunamadı.")
        return []

    host_ids = df['gaia_dr3_id'].dropna().apply(
        lambda x: str(x).replace("Gaia DR3 ", "").split(" ")[-1]
    ).unique().tolist()
    
    print(f"✅ TOPLAM {len(host_ids)} adet Sıcak Jüpiter barındıran sistem analiz edilecek.")
    return host_ids

def calculate_cpm(row1, row2):
    """CPM İndeksi hesaplar."""
    if not (row1['pmra'] and row2['pmra'] and row1['pmdec'] and row2['pmdec']):
        return None, 0

    # Veriler (mas/yr)
    pmRA1 = ufloat(row1['pmra'], row1['pmra_error'])
    pmRA2 = ufloat(row2['pmra'], row2['pmra_error'])
    pmDEC1 = ufloat(row1['pmdec'], row1['pmdec_error'])
    pmDEC2 = ufloat(row2['pmdec'], row2['pmdec_error'])

    # Toplam Öz Hareket
    pm1 = umath.sqrt(pmRA1**2 + pmDEC1**2)
    pm2 = umath.sqrt(pmRA2**2 + pmDEC2**2)
    
    try:
        cpm = (pm1 + pm2) / umath.fabs(pm1 - pm2)
        return cpm.nominal_value, cpm.std_dev
    except ZeroDivisionError:
        return np.inf, 0.0
    except:
        return None, 0

def main():
    # 1. Listeyi Al
    host_ids = get_all_hot_jupiter_hosts()
    if not host_ids: return

    # 2. Dosya Başlığını Yaz
    print(f"💾 Sonuçlar '{OUTPUT_FILE}' dosyasına yazılacak...")
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Host_Gaia_ID', 'Host_Name', 'Candidate_ID', 
            'Ang_Dist_arcsec', 'CPM_Index', 'CPM_Error', 'Status'
        ])

    # 3. Döngü (Tüm Yıldızlar)
    total = len(host_ids)
    start_time = time.time()

    for i, host_id in enumerate(host_ids):
        # İlerleme Çubuğu Benzeri Bilgi
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (total - i - 1)
        print(f"⚙️ İşleniyor: {i+1}/{total} (ID: {host_id}) - Tahmini Kalan Süre: {remaining/60:.1f} dk")

        try:
            # A. Ana Yıldız Verisi
            q_host = f"""SELECT TOP 1 DESIGNATION, ra, dec, pmra, pmra_error, pmdec, pmdec_error 
                         FROM gaiadr3.gaia_source WHERE SOURCE_ID = {host_id}"""
            j_host = Gaia.launch_job(q_host)
            r_host = j_host.get_results()
            
            if len(r_host) == 0: continue
            host_data = r_host[0]

            # B. Yoldaş Adayları
            q_comp = f"""SELECT DESIGNATION, pmra, pmra_error, pmdec, pmdec_error,
                         DISTANCE(POINT({host_data['ra']}, {host_data['dec']}), POINT(ra, dec)) * 3600 AS dist
                         FROM gaiadr3.gaia_source 
                         WHERE 1=CONTAINS(POINT(ra, dec), CIRCLE({host_data['ra']}, {host_data['dec']}, {SEARCH_RADIUS_ARCSEC}/3600.))
                         AND SOURCE_ID != {host_id}"""
            
            j_comp = Gaia.launch_job(q_comp)
            comps = j_comp.get_results()

            if len(comps) == 0: continue

            # C. Analiz ve Kayıt
            candidates_found = 0
            with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                for comp in comps:
                    cpm, err = calculate_cpm(host_data, comp)
                    
                    status = "Bagli Degil"
                    if cpm is not None:
                        if cpm == np.inf: status = "Mukemmel (Esit PM)"
                        elif cpm > CPM_THRESHOLD_GREAT: status = "Mukemmel (>10)"
                        elif cpm > CPM_THRESHOLD_GOOD: status = "Iyi (>3)"
                    
                    # Sadece "İyi" veya "Mükemmel" adayları mı kaydetmek istersiniz?
                    # Şu an Hepsini kaydediyorum, filtrelemeyi Excel'de yapabilirsiniz.
                    
                    writer.writerow([
                        host_id, host_data['DESIGNATION'], comp['DESIGNATION'],
                        f"{comp['dist']:.4f}", 
                        f"{cpm:.2f}" if cpm else "N/A",
                        f"{err:.2f}" if cpm else "0",
                        status
                    ])
                    candidates_found += 1
            
            if candidates_found > 0:
                print(f"   >>> {candidates_found} aday bulundu ve kaydedildi.")

        except Exception as e:
            print(f"   ⚠️ Hata oluştu (ID: {host_id}): {e}")
            continue # Bir yıldız hatalıysa diğerine geç

    print(f"\n✅ İŞLEM TAMAMLANDI. Tüm veriler '{OUTPUT_FILE}' dosyasına kaydedildi.")

if __name__ == '__main__':
    main()