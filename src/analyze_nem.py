#!/usr/bin/env python3
# GridAI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Styles
plt.style.use('seaborn-v0_8-whitegrid')
COLOR = {
    'primary':'#2E4057','secondary':'#048A81','accent':'#D6365F',
    'warning':'#F18F01','success':'#70AE6E','neutral':'#7D8491'
}
REPORTS = Path("../reports"); REPORTS.mkdir(exist_ok=True)
DATA = Path("../data/processed")

# Map AEMO region IDs to names
REGION_LABELS = {'NSW1':'NSW','VIC1':'VIC','QLD1':'QLD','SA1':'SA','TAS1':'TAS'}

# Data prep
def load_data():
    panel = pd.read_csv(DATA/"panel_nem_30min_5yrs.csv", parse_dates=["interval_start"])
    regional = pd.read_csv(DATA/"demand_region_30min.csv",  parse_dates=["interval_start"])
    for df in (panel, regional):
        if hasattr(df['interval_start'].dt, 'tz'):
            df['interval_start'] = df['interval_start'].dt.tz_localize(None)
    if "GEN_MW" not in panel.columns and "NEM_GEN_ENERGY_MWH" in panel.columns:
        panel["GEN_MW"] = panel["NEM_GEN_ENERGY_MWH"] * 2.0
    if "NEM_DEMAND_MW" in panel.columns:
        panel.rename(columns={"NEM_DEMAND_MW":"DEMAND_MW"}, inplace=True)
    panel = panel[(panel["DEMAND_MW"]>0) & (panel["GEN_MW"]>0)]

    # Regional stats
    regional_stats = regional.groupby('REGIONID')['DEMAND_MW'].agg(['mean','std'])
    regional_stats['volatility'] = regional_stats['std'] / regional_stats['mean']
    # add label column
    regional_stats['label'] = [REGION_LABELS.get(r, r.rstrip('0123456789')) for r in regional_stats.index]

    return panel.sort_values("interval_start"), regional_stats

def engineer_features(df):
    df = df.copy(); t = df["interval_start"]
    df["Hour"] = t.dt.hour; df["Day of Week"] = t.dt.dayofweek; df["Month"] = t.dt.month; df["Year"] = t.dt.year
    df["Hour (sin)"] = np.sin(2*np.pi*df["Hour"]/24); df["Hour (cos)"] = np.cos(2*np.pi*df["Hour"]/24)
    df["Weekend"] = (df["Day of Week"]>=5).astype(int)
    df["Peak Hours"] = ((df["Hour"]>=17)&(df["Hour"]<=21)).astype(int)
    df["Summer"] = df["Month"].isin([12,1,2]).astype(int)
    for lag in [1,48,336]:
        df[f"Demand Lag {lag}"] = df["DEMAND_MW"].shift(lag)
    df["24hr Average"] = df["DEMAND_MW"].rolling(48).mean().shift(1)
    df["24hr Volatility"] = df["DEMAND_MW"].rolling(48).std().shift(1)
    years_since_2020 = (df["Year"]-2020).clip(lower=0)
    df["DC Base Load"] = 200*(1.4**years_since_2020)
    df["DC With Cooling"] = df["DC Base Load"]*(1+0.3*df["Summer"])
    df["DC Total Load"] = df["DC With Cooling"]*(1+0.2*df["Peak Hours"])
    df["Supply Margin"] = df["GEN_MW"]-df["DEMAND_MW"]
    df["Reserve (%)"] = (df["Supply Margin"]/df["DEMAND_MW"])*100
    return df.dropna()

# Modelling
def train_models(df):
    feats = ["Hour","Day of Week","Month","Hour (sin)","Hour (cos)","Weekend","Peak Hours","Summer",
             "Demand Lag 1","Demand Lag 48","24hr Average","DC Total Load","Reserve (%)"]
    train = df[df["interval_start"] < "2024-01-01"]; test = df[df["interval_start"] >= "2024-01-01"]
    Xtr,ytr = train[feats], train["DEMAND_MW"]; Xte,yte = test[feats], test["DEMAND_MW"]
    xgb = XGBRegressor(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.8,random_state=42,n_jobs=-1).fit(Xtr,ytr)
    rf  = RandomForestRegressor(n_estimators=200,max_depth=20,random_state=42,n_jobs=-1).fit(Xtr,ytr)
    xgb_pred, rf_pred = xgb.predict(Xte), rf.predict(Xte)
    return {
        'test': test, 'y': yte, 'xgb_pred': xgb_pred, 'rf_pred': rf_pred,
        'xgb_r2': r2_score(yte,xgb_pred), 'rf_r2': r2_score(yte,rf_pred),
        'xgb_rmse': np.sqrt(mean_squared_error(yte,xgb_pred)),
        'rf_rmse':  np.sqrt(mean_squared_error(yte,rf_pred)),
        'imp': pd.DataFrame({'Feature':feats,'Importance':xgb.feature_importances_}).sort_values('Importance',ascending=False)
    }

def savefig(path): plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()

# PLOTS
# 1) Exponential Growth Threatens Grid Stability
def plot_exponential_growth(df):
    monthly = df.set_index('interval_start').resample('M').mean()
    plt.figure(figsize=(8,5))
    plt.plot(monthly.index, monthly['DEMAND_MW'], lw=2.2, color=COLOR['primary'], label='Historical Demand')
    plt.plot(monthly.index, monthly['DEMAND_MW']+monthly['DC Total Load'], lw=2.2, ls='--', color=COLOR['accent'], label='Projected with AI/Data Centers')
    plt.fill_between(monthly.index, monthly['DEMAND_MW'], monthly['DEMAND_MW']+monthly['DC Total Load'], alpha=0.3, color=COLOR['accent'])
    plt.xlabel('Time'); plt.ylabel('Electricity Demand (MW)'); plt.title('Exponential Growth Threatens Grid Stability', fontweight='bold')
    plt.legend(fancybox=True, frameon=True, shadow=True); plt.grid(alpha=0.3)
    savefig(REPORTS/"01_exponential_growth_threatens_grid_stability.png")

# 2) Regional Risk Assessment (Average vs Volatility as bars)
def plot_regional_risk(regional_stats):
    labels = regional_stats['label'].values
    means = (regional_stats['mean']/1000).values
    vols  = (regional_stats['volatility']*100).values
    x = np.arange(len(labels)); w = 0.38
    plt.figure(figsize=(8,5))
    plt.bar(x-w/2, means, w, label='Average Demand (GW)', color=COLOR['secondary'], alpha=0.85)
    plt.bar(x+w/2, vols,  w, label='Volatility (%)',     color=COLOR['warning'],  alpha=0.85)
    plt.xticks(x, labels); plt.xlabel('Region'); plt.ylabel('Value'); plt.title('Regional Risk Assessment', fontweight='bold')
    plt.grid(axis='y', alpha=0.3); plt.legend(fancybox=True, frameon=True, shadow=True)
    savefig(REPORTS/"02_regional_risk_assessment.png")

# 3) Model Performance Metrics (R2 + RMSE)
def plot_model_performance_metrics(res):
    models=['XGBoost','Random Forest']; x=np.arange(2); w=0.35
    plt.figure(figsize=(8,5))
    ax=plt.gca(); ax2=ax.twinx()
    b1=ax.bar(x-w/2,[res['xgb_r2'],res['rf_r2']], w, label='R²',   color=COLOR['success'], alpha=0.85)
    _ = ax2.bar(x+w/2,[res['xgb_rmse'],res['rf_rmse']], w, label='RMSE', color=COLOR['warning'], alpha=0.85)
    ax.set_ylim(0.9,1.0); ax.set_ylabel('R²', color=COLOR['success']); ax2.set_ylabel('RMSE (MW)', color=COLOR['warning'])
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_xlabel('Model'); ax.grid(alpha=0.3)
    plt.title('Model Performance Metrics', fontweight='bold')
    for b,v in zip(b1,[res['xgb_r2'],res['rf_r2']]): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.004,f"{v:.3f}",ha='center')
    savefig(REPORTS/"03_model_performance_metrics.png")

# 4) Key Demand Drivers (XGB feature importance)
def plot_key_demand_drivers(res):
    top = res['imp'].head(8)
    colors=[COLOR['primary'] if 'DC' in f else COLOR['secondary'] for f in top['Feature']]
    plt.figure(figsize=(8,5))
    plt.barh(range(len(top)), top['Importance'], color=colors, alpha=0.85)
    plt.yticks(range(len(top)), top['Feature']); plt.xlabel('Importance Score')
    plt.title('Key Demand Drivers', fontweight='bold'); plt.grid(axis='x', alpha=0.3)
    savefig(REPORTS/"04_key_demand_drivers.png")

# 5) XGBoost & Random Forest Accuracy
def plot_accuracy_side_by_side(res):
    sample=min(2000, len(res['y'])); idx=np.random.choice(len(res['y']), sample, replace=False)
    y=res['y'].iloc[idx]; mn, mx = res['y'].min(), res['y'].max()
    fig,ax = plt.subplots(1,2,figsize=(12,5))
    ax[0].scatter(y, res['xgb_pred'][idx], s=10, alpha=0.5, color=COLOR['secondary'])
    ax[0].plot([mn,mx],[mn,mx],'r--',lw=2,alpha=0.7)
    ax[0].set_title(f"XGBoost Accuracy (R²={res['xgb_r2']:.3f})"); ax[0].set_xlabel('Actual (MW)'); ax[0].set_ylabel('Predicted (MW)'); ax[0].grid(alpha=0.3)
    ax[1].scatter(y, res['rf_pred'][idx],  s=10, alpha=0.5, color=COLOR['success'])
    ax[1].plot([mn,mx],[mn,mx],'r--',lw=2,alpha=0.7)
    ax[1].set_title(f"Random Forest Accuracy (R²={res['rf_r2']:.3f})"); ax[1].set_xlabel('Actual (MW)'); ax[1].set_ylabel('Predicted (MW)'); ax[1].grid(alpha=0.3)
    savefig(REPORTS/"05_accuracy_xgb_vs_rf_side_by_side.png")

# 6) Infrastructure Gap by Growth Scenario
def plot_infrastructure_gap(res):
    scenarios=['Current','+10%','+20%','+35%','+50%']; mult=[1.0,1.1,1.2,1.35,1.5]
    xm, rm = [], []; gen = res['test']['GEN_MW'].values
    for m in mult:
        xm.append(np.maximum(0, res['xgb_pred']*m - gen).mean())
        rm.append(np.maximum(0, res['rf_pred'] *m - gen).mean())
    x=np.arange(len(scenarios)); w=0.35
    plt.figure(figsize=(8,5))
    plt.bar(x-w/2, xm, w, label='XGBoost Projection', color=COLOR['secondary'], alpha=0.85)
    plt.bar(x+w/2, rm, w, label='Random Forest Projection', color=COLOR['success'],  alpha=0.85)
    plt.axhline(500, ls='--', color=COLOR['accent'], lw=2, label='Illustrative Critical Threshold')
    plt.xticks(x, scenarios); plt.xlabel('Demand Growth Scenario'); plt.ylabel('Avg Supply Deficit (MW)')
    plt.title('Infrastructure Gap by Growth Scenario (Model-based)', fontweight='bold'); plt.grid(axis='y', alpha=0.3); plt.legend(fancybox=True, frameon=True, shadow=True)
    savefig(REPORTS/"06_infrastructure_gap_by_growth_scenario.png")


# 7) Regional Demand Volatility Scatter with clean labels
def plot_regional_demand_vol_scatter(regional_stats):
    labels = regional_stats['label'].values
    x = (regional_stats['mean']/1000).values
    y = (regional_stats['volatility']*100).values
    plt.figure(figsize=(8,5))
    plt.scatter(x, y, s=200, alpha=0.6, color=COLOR['secondary'])
    for lab, xv, yv in zip(labels, x, y):
        plt.annotate(lab, (xv, yv), ha='center', va='center', fontsize=10, fontweight='bold')
    plt.xlabel('Average Demand (GW)'); plt.ylabel('Volatility (%)')
    plt.title('Regional Demand vs Volatility', fontweight='bold'); plt.grid(alpha=0.3)
    savefig(REPORTS/"08_regional_demand_vs_volatility.png")

# 8) Declining Safety Margins
def plot_declining_margins(df):
    # Monthly average reserve margin
    margins_series = df.groupby(df['interval_start'].dt.to_period('M'))['Reserve (%)'].mean()
    values = margins_series.values
    x = np.arange(len(values))

    plt.figure(figsize=(10,5))
    # shaded critical zone up to 10%
    plt.fill_between(x, 0, 10, alpha=0.15, color=COLOR['accent'])
    # dashed reference line at 10%
    plt.axhline(10, ls='--', color=COLOR['accent'], lw=2, label='Critical Threshold (10%)')

    plt.plot(x, values, lw=2.4, color=COLOR['primary'], marker='o', ms=4)

    # dates corresponding to each monthly point
    dates = margins_series.index.to_timestamp()  # PeriodIndex to Timestamp
    x = np.arange(len(values))

    # choose a spacing (every 8 labels)
    step = max(1, len(dates) // 8)

    # sparse ticks
    plt.xticks(
        ticks=np.arange(0, len(dates), step),
        labels=[d.strftime('%Y-%m') for d in dates[::step]],
        rotation=45, ha='right'
    )
    plt.subplots_adjust(bottom=0.18) # leave space

    # annotate the largest drop including the reasons
    min_idx = int(np.argmin(values))
    min_val = float(values[min_idx])

    annotation = (
        "Why a sharp drop?\n"
        "  • Generator outages/deratings\n"
        "  • Extreme weather (heatwave/cold snap)\n"
        "  • Interconnector limits or maintenance\n"
        "  • Demand spike + low VRE output"
    )
    plt.annotate(
        annotation,
        xy=(min_idx, min_val),
        xytext=(min_idx + max(2, len(values)//20), min_val + 4),
        textcoords='data',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor=COLOR['neutral']),
        arrowprops=dict(arrowstyle='->', color=COLOR['primary'], lw=1.5),
        ha='left', va='bottom'
    )

    plt.xlabel('Month (labels omitted for clarity)'); plt.ylabel('Reserve Margin (%)')
    plt.title('Declining Safety Margins', fontweight='bold'); plt.grid(alpha=0.25, axis='y')
    plt.legend()
    savefig(REPORTS/"09_declining_safety_margins.png")

def main():
    df, regional_stats = load_data()
    df = engineer_features(df)
    res = train_models(df)
    print(f"XGB R²={res['xgb_r2']:.3f} RMSE={res['xgb_rmse']:.1f} | RF R²={res['rf_r2']:.3f} RMSE={res['rf_rmse']:.1f}")

    # Plots
    plot_exponential_growth(df)
    plot_regional_risk(regional_stats)
    plot_model_performance_metrics(res)
    plot_key_demand_drivers(res)
    plot_accuracy_side_by_side(res)
    plot_infrastructure_gap(res)
    plot_regional_demand_vol_scatter(regional_stats)
    plot_declining_margins(df)

if __name__ == "__main__":
    main()