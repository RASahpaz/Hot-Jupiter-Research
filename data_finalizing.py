#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 12:37:10 2026

@author: resulayberksahpaz
"""
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.gaia import Gaia
import pandas as pd
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from uncertainties import ufloat
from uncertainties.umath import sqrt as usqrt

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

print("🚀 Pipeline başlatılıyor...")
print("------------------------------------------------------------")

# ---------------- Parallax fonksiyonları ----------------
def parallax_distance(p, p_err):
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

# ---------------- NASA GAIA sütunu tespiti ----------------
print("🔭 NASA Exoplanet Archive sorgulanıyor...")
tbl = NasaExoplanetArchive.query_criteria(table="pscomppars", select="top 1 *")
colnames = list(tbl.colnames)
gaia_col = "gaia_dr3_id" if "gaia_dr3_id" in colnames else None
print(f"➡️ GAIA DR3 kolonu: {gaia_col}")

# ---------------- Sıcak Jüpiterleri çek ----------------
print("🔥 Hot Jupiter sistemleri çekiliyor...")

select_fields = ["pl_name","hostname","pl_orbper","pl_bmassj","ra","dec"]
if gaia_col:
    select_fields.append(gaia_col)

hot_jupiters = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select=",".join(select_fields),
    where="pl_bmassj > 0.3 AND pl_orbper < 10"
).to_pandas()
hot_jupiters = hot_jupiters.dropna(subset=["ra","dec"])

print(f"➡️ Bulunan Hot Jupiter sayısı: {len(hot_jupiters)}")
print("------------------------------------------------------------")

# ---------------- Gaia yakın kaynak araması ----------------
def search_gaia_nearby(ra, dec, radius_arcsec=5):
    radius_deg = radius_arcsec / 3600.0
    print(f"   🔍 Gaia yakın kaynak araması: RA={ra:.5f}, DEC={dec:.5f}")
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

# ---------------- Açısal ayrılık ----------------
def angular_sep(ra1, dec1, ra2, dec2):
    c1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
    c2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
    return c1.separation(c2).arcsecond

# ---------------- CPM metrikleri ----------------
def simple_pm_diff(pmra1, pmdec1, pmra2, pmdec2):
    if any(pd.isna([pmra1, pmdec1, pmra2, pmdec2])):
        return np.nan
    return np.sqrt((pmra1-pmra2)**2 + (pmdec1-pmdec2)**2)

def mugrauer_cpm(pmra1, pmra1_err, pmdec1, pmdec1_err,
                 pmra2, pmra2_err, pmdec2, pmdec2_err):
    vals = [pmra1,pmra1_err,pmdec1,pmdec1_err,pmra2,pmra2_err,pmdec2,pmdec2_err]
    if any(pd.isna(vals)):
        return None,None
    try:
        p1 = usqrt(ufloat(pmra1,pmra1_err)**2 + ufloat(pmdec1,pmdec1_err)**2)
        p2 = usqrt(ufloat(pmra2,pmra2_err)**2 + ufloat(pmdec2,pmdec2_err)**2)
        denom = abs(p1 - p2)
        if denom.nominal_value==0: return None,None
        cpm = (p1+p2)/denom
        return cpm.nominal_value, cpm.std_dev
    except:
        return None,None

# ---------------- Candidate Gaia DR3 ID kontrolü ----------------
def get_candidate_dr3_id(candidate_source_id):
    candidate_source_id = int(candidate_source_id)
    print(f"      📡 Gaia DR3 ID kontrolü: {candidate_source_id}")
    try:
        query = f"SELECT source_id FROM gaiadr3.gaia_source WHERE source_id = {candidate_source_id}"
        job = Gaia.launch_job_async(query, dump_to_file=False)
        res = job.get_results()
        if len(res) > 0:
            return int(res[0]["source_id"])
        else:
            return "Gaia arşivinde yok"
    except:
        return "Gaia arşivinde yok"

# ---------------- Ana pipeline ----------------
results = []
no_nearby_list = []   # <--- YENİ EKLENDİ

print("🌌 Companion araması başlıyor...")
print("------------------------------------------------------------")

for idx,row in hot_jupiters.iterrows():
    hostname = row.get("hostname","")
    pl_name = row.get("pl_name","")
    print(f"\n⭐ {idx+1}/{len(hot_jupiters)} → {hostname} / {pl_name}")

    try:
        nearby = search_gaia_nearby(row["ra"],row["dec"],radius_arcsec=5)
    except Exception as e:
        print(f"   ❌ Gaia sorgusu başarısız: {e}")
        continue

    print(f"   ➡️ Yakın kaynak sayısı: {len(nearby)}")

    if len(nearby)<=1:
        print("   ⚠️ Bu sistemde başka Gaia kaynağı yok.")

        no_nearby_list.append({
            "host": hostname,
            "planet": pl_name,
            "ra": row["ra"],
            "dec": row["dec"],
            "gaia_dr3_id": row.get(gaia_col, np.nan),
            "nearby_sources": 0
        })
        continue

    nearby_count = len(nearby)-1

    # Primary seçimi
    primary = None
    if gaia_col and gaia_col in row and not pd.isna(row[gaia_col]):
        try:
            gid = int(row[gaia_col])
            match = nearby.loc[nearby["source_id"]==gid]
            if len(match)>0: 
                primary = match.iloc[0]
                print(f"   🌐 Primary GAIA eşleşti: {gid}")
        except:
            primary=None

    if primary is None:
        primary = nearby.sort_values("phot_g_mean_mag").iloc[0]
        print(f"   🌐 Primary seçildi (en parlak yıldız): {int(primary['source_id'])}")

    primary_dr3_id = row.get(gaia_col,np.nan)

    # Candidate loop
    for _,cand in nearby.iterrows():
        if int(cand["source_id"])==int(primary["source_id"]):
            continue

        print(f"   ➕ Aday inceleniyor: {int(cand['source_id'])}")

        sep_arcsec = angular_sep(primary["ra"],primary["dec"],cand["ra"],cand["dec"])
        pm_diff = simple_pm_diff(primary["pmra"],primary["pmdec"],cand["pmra"],cand["pmdec"])
        cpm_val, cpm_err = mugrauer_cpm(
            primary["pmra"],primary["pmra_error"],primary["pmdec"],primary["pmdec_error"],
            cand["pmra"],cand["pmra_error"],cand["pmdec"],cand["pmdec_error"]
        )

        dist_primary = parallax_distance(primary["parallax"],primary["parallax_error"])
        dist_cand = parallax_distance(cand["parallax"],cand["parallax_error"])
        par_diff, par_diff_err = parallax_difference(
            primary["parallax"],primary["parallax_error"],
            cand["parallax"],cand["parallax_error"]
        )
        par_sigma = parallax_agreement(
            primary["parallax"],primary["parallax_error"],
            cand["parallax"],cand["parallax_error"]
        )

        candidate_dr3_id = get_candidate_dr3_id(cand["source_id"])

        if np.isfinite(pm_diff) and pm_diff < 5:
            print("      ✅ CPM ADAYI BULUNDU!")
            results.append({
                "host": hostname,
                "planet": pl_name,
                "primary_gaia_dr3_id": primary_dr3_id,
                "candidate_gaia_dr3_id": candidate_dr3_id,
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
        else:
            print("      ❌ CPM uyumsuz — aday elendi.")

# ---------------- Sonuçlar ----------------
df = pd.DataFrame(results)
if not df.empty:
    df = df.sort_values(by=["parallax_sigma_agreement","pm_diff_masyr"],na_position="last")

output_file = "hot_jupiter_cpm_parallax_candidates_gaia_dr3.csv"
df.to_csv(output_file,index=False)

# ---- Yakında Gaia kaynağı olmayanlar için ayrı CSV ----
df_no = pd.DataFrame(no_nearby_list)
output_file_no = "hot_jupiter_no_nearby_gaia.csv"
df_no.to_csv(output_file_no, index=False)

print("\n------------------------------------------------------------")
print(f"💾 CPM sonuçları '{output_file}' dosyasına kaydedildi.")
print(f"💾 Yakın Gaia kaynağı olmayan sistemler '{output_file_no}' dosyasına kaydedildi.")
print("🌠 Pipeline tamamlandı — yıldızlar usulca yerine yerleşti.")




from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd
import numpy as np

# ============================================================
# NASA EXOPLANET ARCHIVE'DAN ANA GEZEGEN KATALOĞU
# Mp > 0.3 Mjup, P < 10 gün
# ============================================================

print("⏳ NASA Exoplanet Archive sorgulanıyor...")

exo = NasaExoplanetArchive.query_criteria(
    table="pscomppars",
    select="""
        pl_name, hostname,
        pl_orbper, pl_bmassj,
        ra, dec,
        sy_gaia_id,
        st_pmra, st_pmdec,
        st_plx
    """,
    where="""
        pl_bmassj > 0.3
        AND pl_orbper < 10
        AND sy_gaia_id IS NOT NULL
        AND st_pmra IS NOT NULL
        AND st_pmdec IS NOT NULL
        AND st_plx IS NOT NULL
    """,
    cache=True
)

exo = exo.to_pandas()

# ------------------------------------------------------------
# KOLONLARI SANA UYGUN HALE GETİR
# ------------------------------------------------------------

exo = exo.rename(columns={
    "sy_gaia_id": "gaia_id",
    "st_pmra": "pmra",
    "st_pmdec": "pmdec",
    "st_plx": "parallax"
})

# Gaia ID bazen 'Gaia DR3 123456...' şeklinde geliyor
exo["gaia_id"] = (
    exo["gaia_id"]
    .astype(str)
    .str.replace("Gaia DR3 ", "", regex=False)
    .astype("int64")
)

exo = exo.dropna(subset=[
    "ra", "dec", "pmra", "pmdec", "parallax"
])

print(f"✔ Seçilen gezegen sayısı: {len(exo)}")
print(f"✔ Sistem sayısı: {exo['hostname'].nunique()}")

# ------------------------------------------------------------
# İSTERSEN ARA ÇIKTI
# ------------------------------------------------------------

exo.to_csv("hot_jupiters.csv", index=False)



# ============================================================
# POST-PROCESSING: SYSTEM-LEVEL SUMMARY & CROSS-MATCH
# Bu blok, ana companion arama kodundan SONRA çalışır
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. DOSYALARI YÜKLE
# ------------------------------------------------------------

# Senin kodunun ürettiği companion çıktısı
COMPANION_FILE = "hot_jupiter_cpm_parallax_candidates_gaia_dr3.csv"        # <-- senin dosya adın
EXOPLANET_FILE = "hot_jupiters.csv"       # ana gezegen kataloğu
EXTERNAL_CATALOG = "companion_checked_hot_jupiters.csv"  # cross-match yapılacak katalog

companions_df = pd.read_csv(COMPANION_FILE)
exo_df = pd.read_csv(EXOPLANET_FILE)

# ------------------------------------------------------------
# 2. SİSTEM BAZLI GEZEGEN ÖZETİ
# ------------------------------------------------------------

planet_summary = (
    exo_df
    .groupby("hostname")
    .agg(
        n_planets=("pl_name", "nunique"),
        planets=("pl_name", lambda x: "; ".join(sorted(x.unique()))),
        host_gaia_id=("gaia_id", "first")
    )
    .reset_index()
)

# ------------------------------------------------------------
# 3. COMPANION ÖZETİ (SYSTEM LEVEL)
# ------------------------------------------------------------

if not companions_df.empty:
    companion_summary = (
        companions_df
        .groupby("hostname")
        .agg(
            n_companions=("companion_gaia_id", "nunique"),
            companion_gaia_ids=(
                "companion_gaia_id",
                lambda x: "; ".join(map(str, sorted(x.unique())))
            )
        )
        .reset_index()
    )
else:
    companion_summary = pd.DataFrame(
        columns=["hostname", "n_companions", "companion_gaia_ids"]
    )

systems_df = planet_summary.merge(
    companion_summary,
    on="hostname",
    how="left"
)

systems_df["n_companions"] = systems_df["n_companions"].fillna(0).astype(int)
systems_df["companion_gaia_ids"] = systems_df["companion_gaia_ids"].fillna("")

# ------------------------------------------------------------
# 4. COMPANIONLARI TEK SATIRDA (WIDE FORMAT)
# ------------------------------------------------------------

if not companions_df.empty:
    companions_wide = (
        companions_df
        .assign(idx=lambda df: df.groupby("hostname").cumcount() + 1)
        .pivot(index="hostname", columns="idx", values="companion_gaia_id")
        .reset_index()
    )

    companions_wide.columns = [
        "hostname" if c == "hostname" else f"companion_{c}_gaia_id"
        for c in companions_wide.columns
    ]

    systems_df = systems_df.merge(
        companions_wide,
        on="hostname",
        how="left"
    )

# ------------------------------------------------------------
# 5. EXTERNAL CATALOG CROSS-MATCH
# ------------------------------------------------------------

external_df = pd.read_csv(EXTERNAL_CATALOG)

systems_df["in_external_catalog"] = (
    systems_df["hostname"].isin(external_df["hostname"])
)

final_df = systems_df.merge(
    external_df,
    on="hostname",
    how="left",
    suffixes=("", "_external")
)

# ------------------------------------------------------------
# 6. ÇIKTILAR
# ------------------------------------------------------------

systems_df.to_csv(
    "systems_summary.csv",
    index=False
)

final_df.to_csv(
    "systems_crossmatched.csv",
    index=False
)

print("✔ Post-processing tamamlandı.")
print(f"✔ Toplam sistem: {len(systems_df)}")
print(f"✔ Companion bulunan sistem: {(systems_df['n_companions'] > 0).sum()}")

