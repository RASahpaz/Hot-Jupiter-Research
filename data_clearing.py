#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:05:18 2025

@author: resulayberksahpaz
"""

import pandas as pd

# Load your data
df = pd.read_csv("hot_jupiter_cpm_parallax_candidates_gaia_dr3.csv")

# If the error column has a specific name, replace "mugrauer_cpm_err"
df["cpm_min"] = df["mugrauer_cpm"] - df["mugrauer_cpm_err"]

# Condition 1: mugrauer_cpm >= 3
cond_nominal = df["mugrauer_cpm"] >= 3

# Condition 2a: Even with error, still >= 3 → "robust"
cond_robust = df["cpm_min"] >= 3

# Condition 2b: Nominal >= 3 but error pushes below 3 → "fragile"
cond_fragile = (df["mugrauer_cpm"] >= 3) & (df["cpm_min"] < 3)

# Create outputs
df_robust = df[cond_nominal & cond_robust]
df_fragile = df[cond_fragile]

# Save them
df_robust.to_csv("mugrauer_cpm_robust_over3.csv", index=False)
df_fragile.to_csv("mugrauer_cpm_fragile_over3.csv", index=False)

print("Done! Two files created:")
print(" - mugrauer_cpm_robust_over3.csv  (error can't drop it below 3)")
print(" - mugrauer_cpm_fragile_over3.csv (error CAN drop it below 3)")
