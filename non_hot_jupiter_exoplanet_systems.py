#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 11:58:34 2025
Author: Resul Ayberk Şahpaz
Description: Identify planetary systems that *do have planets* but *do NOT host hot Jupiters*
"""

from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd
import numpy as np

print("🔭 Tüm gezegenli sistemler çekiliyor...")

# --------------------------------------------------------------
# 1. Bütün gezegenli sistemleri çek
# --------------------------------------------------------------
tbl = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="hostname, pl_name, pl_bmassj, pl_orbper"
)

df = tbl.to_pandas()

print(f"→ Toplam gezegen girdisi: {len(df)}")
print(f"→ Farklı yıldız sistemi: {df['hostname'].nunique()}")

# --------------------------------------------------------------
# 2. Hot Jupiter tanımı
# --------------------------------------------------------------
def is_hot_jupiter(row):
    if pd.isna(row["pl_bmassj"]) or pd.isna(row["pl_orbper"]):
        return False
    return (row["pl_bmassj"] > 0.3) and (row["pl_orbper"] < 10)

df["is_hot_jup"] = df.apply(is_hot_jupiter, axis=1)

# --------------------------------------------------------------
# 3. Sistemleri grupla → hot Jupiter var mı yok mu?
# --------------------------------------------------------------
system_summary = (
    df.groupby("hostname")["is_hot_jup"]
      .any()  # en az bir sıcak Jüpiter varsa True
      .reset_index()
      .rename(columns={"is_hot_jup": "has_hot_jupiter"})
)

# --------------------------------------------------------------
# 4. Hot Jupiter BARINDIRMAYAN sistemleri çek
# --------------------------------------------------------------
cold_systems = system_summary[system_summary["has_hot_jupiter"] == False]

print(f"\n🌙 Hot Jupiter içermeyen sistem sayısı: {len(cold_systems)}")

# --------------------------------------------------------------
# 5. Bu sistemlere ait gezegenleri de ekleyelim (opsiyonel)
# --------------------------------------------------------------
merged = cold_systems.merge(df, on="hostname", how="left")

# --------------------------------------------------------------
# 6. Kaydet
# --------------------------------------------------------------
output_file = "systems_without_hot_jupiters.csv"
merged.to_csv(output_file, index=False)

print(f"\n💾 Sonuçlar '{output_file}' dosyasına kaydedildi.")
print("✨ İlk 10 satır:")
print(merged.head(10))
