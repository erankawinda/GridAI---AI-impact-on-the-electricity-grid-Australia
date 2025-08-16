import pathlib as pl
import pandas as pd

CACHE = pl.Path.home() / "nem-data" / "data"
OUT = pl.Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def load_parquets(root: pl.Path):
    files = sorted(root.rglob("clean.parquet"))
    if not files:
        raise SystemExit(f"No clean.parquet under {root}. Did you run nemdata?")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

# DEMAND (region x 30-min)
print("Loading demand…")
dfd = load_parquets(CACHE / "demand")
cols_l = {c.lower(): c for c in dfd.columns}
interval_col = cols_l.get("interval-start", "interval-start")
if interval_col not in dfd.columns:
    raise ValueError(f"interval-start not found in demand columns: {dfd.columns.tolist()}")
region_col = "REGIONID" if "REGIONID" in dfd.columns else ("REGION" if "REGION" in dfd.columns else None)
if region_col is None:
    raise ValueError(f"REGIONID/REGION not found in demand columns: {dfd.columns.tolist()}")
val_col = "DEMAND" if "DEMAND" in dfd.columns else ("TOTALDEMAND" if "TOTALDEMAND" in dfd.columns else None)
if val_col is None:
    raise ValueError(f"DEMAND/TOTALDEMAND not found in demand columns: {dfd.columns.tolist()}")

demand_30 = (
    dfd[[interval_col, region_col, val_col]]
    .rename(columns={interval_col:"interval_start", region_col:"REGIONID", val_col:"DEMAND_MW"})
    .sort_values(["interval_start", "REGIONID"])
)
nem_demand_30 = (demand_30.groupby("interval_start", as_index=False)["DEMAND_MW"]
                 .sum().rename(columns={"DEMAND_MW":"NEM_DEMAND_MW"}))

# GENERATION (unit-scada 5-min -> 30-min, corrected aggregation)
dfs = load_parquets(CACHE / "unit-scada")
dfs.columns = [c.upper() for c in dfs.columns]
for req in ("SETTLEMENTDATE","SCADAVALUE"):
    if req not in dfs.columns:
        raise ValueError(f"Missing {req} in unit-scada columns: {dfs.columns.tolist()}")

# 1) Sum output across all units for each 5-min interval
scada_5 = (dfs.groupby("SETTLEMENTDATE", as_index=False)["SCADAVALUE"]
             .sum()
             .rename(columns={"SCADAVALUE":"TOTAL_MW"}))

# 2) Compute energy per 5-min and roll to 30-min
scada_5["ENERGY_MWH_5MIN"] = scada_5["TOTAL_MW"] * (5.0/60.0)
scada_5["HALF_HOUR"] = pd.to_datetime(scada_5["SETTLEMENTDATE"]).dt.floor("30min")

nem_gen_30 = (scada_5.groupby("HALF_HOUR", as_index=False)
                .agg(NEM_GEN_ENERGY_MWH=("ENERGY_MWH_5MIN","sum"),
                     NEM_GEN_AVG_MW=("TOTAL_MW","mean"))
                .rename(columns={"HALF_HOUR":"interval_start"}))
# Align to common window: last 5 full years
start_d, end_d = demand_30["interval_start"].min(), demand_30["interval_start"].max()
start_g, end_g = nem_gen_30["interval_start"].min(), nem_gen_30["interval_start"].max()
common_start = max(start_d, start_g)
common_end   = min(end_d, end_g)
five_years_ago = (pd.to_datetime(common_end) - pd.DateOffset(years=5)).floor("D")
clip_start = max(common_start, five_years_ago)

demand_30 = demand_30.query(" @clip_start <= interval_start <= @common_end ").copy()
nem_demand_30 = nem_demand_30.query(" @clip_start <= interval_start <= @common_end ").copy()
nem_gen_30 = nem_gen_30.query(" @clip_start <= interval_start <= @common_end ").copy()

# Save results
demand_30.to_csv(OUT / "demand_region_30min.csv", index=False)
nem_demand_30.to_csv(OUT / "demand_nem_30min.csv", index=False)
nem_gen_30.to_csv(OUT / "generation_nem_30min.csv", index=False)

panel = (nem_demand_30.merge(nem_gen_30, on="interval_start", how="inner")
         .sort_values("interval_start"))
panel.to_csv(OUT / "panel_nem_30min_5yrs.csv", index=False)

print("Common window:", clip_start, "→", common_end)
for f in ["demand_region_30min.csv","demand_nem_30min.csv","generation_nem_30min.csv","panel_nem_30min_5yrs.csv"]:
    print("Wrote:", OUT / f)
