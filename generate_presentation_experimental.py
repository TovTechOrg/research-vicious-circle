"""
Generate EXPERIMENTAL HTML dashboard presentation.
Light-themed, Chart.js-based alternative to the dark Plotly executive presentation.
Reads benefits_final.csv, trains the same models, loads temporal + distance data,
produces presentation_experimental.html.

Sections:
  1. Header + KPI Cards
  2. What Drives Disability Rates? (OLS coefficients + RF/XGB importance)
  3. Under-Utilization Pockets (Q1)
  4. Intergenerational Trap (Q2)
  5. The Arab Sector Wall
  6. Swords of Iron (Q3)
  7. Distance Experiment (Q4)
  8. Recommendations
"""

from __future__ import annotations

import html as html_lib
import json
import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None
    HAS_XGBOOST = False

try:
    from tabpfn import TabPFNRegressor
    HAS_TABPFN = True
except Exception:
    TabPFNRegressor = None
    HAS_TABPFN = False

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "datas_for_research_vicious_circle_project"
CSV_PATH = DATA_DIR / "data" / "processed" / "benefits_final.csv"


# ── helpers ──────────────────────────────────────────────────────────────────

def fmt_num(val: float, digits: int = 2) -> str:
    if pd.isna(val):
        return ""
    return f"{float(val):.{digits}f}"


def fmt_int(val: float) -> str:
    if pd.isna(val):
        return ""
    return f"{int(round(float(val))):,}"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse_val = mean_squared_error(y_true, y_pred)
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mse_val)), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }


# ── Temporal data helpers ────────────────────────────────────────────────────

def _normalize_name(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = unicodedata.normalize("NFKC", s).strip()
    s = s.replace("\u2013", "-").replace("\u05be", "-").replace("\u05f3", "'").replace("\u05f4", '"')
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False),
        errors="coerce",
    )


def _detect_header_row(path, sheet_name=0, max_rows=160):
    tmp = pd.read_excel(path, sheet_name=sheet_name, header=None)
    patterns = [r"\u05e9\u05dd", r"\u05d9\u05d9\u05e9\u05d5\u05d1", r"\u05e1\u05de\u05dc", r"\u05e1\u05da", r"\u05de\u05e7\u05d1\u05dc\u05d9"]
    best_i, best_score = 0, -1
    for i in range(min(max_rows, len(tmp))):
        row_str = " ".join(tmp.iloc[i].astype(str).tolist())
        score = sum(bool(re.search(p, row_str)) for p in patterns)
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def _read_table(path, sheet_name=0, engine=None):
    h = _detect_header_row(path, sheet_name)
    tbl = pd.read_excel(path, sheet_name=sheet_name, header=[h, h + 1], engine=engine)
    tbl = tbl.dropna(axis=1, how="all")
    return tbl


def _flatten_cols(cols):
    flat = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(x).strip() for x in c if str(x) != "nan"]
            flat.append(" ".join([p for p in parts if p]).strip())
        else:
            flat.append(str(c).strip())
    return flat


def _pick_col(cols, keys):
    for c in cols:
        if any(k in c for k in keys):
            return c
    raise ValueError(f"Column not found for keys {keys}. Example cols: {cols[:40]}")


def _weighted_summary(sub_df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = (
        sub_df.groupby("year", as_index=False)
        .agg(
            total_population=("population", "sum"),
            total_recipients=("recipients", "sum"),
            n_settlements=("settlement_code", "nunique"),
        )
    )
    out["rate"] = 100 * out["total_recipients"] / out["total_population"]
    out["series"] = label
    return out


# ── load data ────────────────────────────────────────────────────────────────

print("Loading data from benefits_final.csv ...")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df)} rows, {len(df.columns)} columns")

numeric_cols = [
    "total_population", "population_0_17", "population_18_64", "population_65_plus",
    "general_disability_benefit", "special_services_for_persons_with_disabilities",
    "mobility_benefit", "income_support_benefit", "long_term_care_benefit",
    "socio_economic_index_score", "socio_economic_index_cluster",
    "peripherality_index_score", "peripherality_index_cluster",
    "edu_attain_pct_academic_degree", "average_monthly_salary_2023",
    "arab_population_percentage", "haredi_population_percentage",
    "jewish_population_percentage", "jewish_non_haredi_population_percentage",
    "disabled_child_benefit_rate",
    "lat", "lon",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Derived rates
df["general_disability_rate"] = (
    df["general_disability_benefit"] / df["population_18_64"] * 100
).round(2)
df["income_support_rate"] = (
    df["income_support_benefit"] / df["population_18_64"] * 100
).round(2)
df["age65_pct"] = (df["population_65_plus"] / df["total_population"] * 100).round(2)
df["age0_17_pct"] = (df["population_0_17"] / df["total_population"] * 100).round(2)
df["log_total_pop"] = np.log1p(pd.to_numeric(df["total_population"], errors="coerce")).round(4)

# Ensure disabled_child_benefit_rate exists
if "disabled_child_benefit_rate" not in df.columns:
    if "disabled_child_benefit" in df.columns and "population_0_17" in df.columns:
        df["disabled_child_benefit_rate"] = (
            pd.to_numeric(df["disabled_child_benefit"], errors="coerce")
            / pd.to_numeric(df["population_0_17"], errors="coerce") * 100
        ).round(2)
    else:
        df["disabled_child_benefit_rate"] = np.nan

# ── Temporal data loading ────────────────────────────────────────────────────

print("Loading frontline temporal data ...")

frontline_codes_df = pd.read_csv(DATA_DIR / "frontline_settlements_codes_swords_of_iron.csv")
frontline_codes = set(
    pd.to_numeric(frontline_codes_df["settlement_code"], errors="coerce").dropna().astype(int)
)
print(f"  {len(frontline_codes)} frontline settlement codes loaded")

# 2024 data
b24_raw = _read_table(DATA_DIR / "benefits_2024_12.xlsx")
b24_raw.columns = _flatten_cols(b24_raw.columns)
col24_name = _pick_col(b24_raw.columns.tolist(), ["\u05e9\u05dd \u05d9\u05d9\u05e9\u05d5\u05d1", "\u05d9\u05d9\u05e9\u05d5\u05d1"])
col24_code = _pick_col(b24_raw.columns.tolist(), ["\u05e1\u05de\u05dc", "\u05e7\u05d5\u05d3"])
col24_pop = _pick_col(b24_raw.columns.tolist(), ["\u05e1\u05da \u05db\u05dc", "\u05ea\u05d5\u05e9\u05d1\u05d9\u05dd"])
col24_dis = _pick_col(b24_raw.columns.tolist(), ["\u05e0\u05db\u05d5\u05ea"])

b24 = pd.DataFrame({
    "settlement_name": b24_raw[col24_name],
    "settlement_code": _to_number(b24_raw[col24_code]).round(0).astype("Int64"),
    "population_2024": _to_number(b24_raw[col24_pop]).round(0).astype("Int64"),
    "disability_recipients_2024": _to_number(b24_raw[col24_dis]),
})
b24["settlement_name"] = b24["settlement_name"].astype(str).str.strip()
b24 = b24.dropna(subset=["settlement_code", "population_2024"])
b24 = b24[b24["population_2024"] > 0]
bad_name_patterns = r'(\u05e1\u05d4"?\u05db|\u05e1\u05d4\u05f3\u05f3\u05db|\u05e1\u05d4\u05f4\u05db|\u05e1\u05da|Total|\u05de\u05d5\u05e2\u05e6\u05d4 \u05d0\u05d6\u05d5\u05e8\u05d9\u05ea|\u05de"\u05d0|\u05de.\u05d0|\u05de\u05d7\u05d5\u05d6|\u05e0\u05e4\u05d4)'
b24 = b24[~b24["settlement_name"].str.contains(bad_name_patterns, regex=True, na=False)]
b24 = b24[b24["settlement_code"] > 0]
b24 = b24.drop_duplicates(subset=["settlement_code"]).copy()
b24["disability_rate_2024"] = (b24["disability_recipients_2024"] / b24["population_2024"]) * 100
b24["name_norm"] = b24["settlement_name"].apply(_normalize_name)

# 2023 data
b23_raw = _read_table(DATA_DIR / "2023-Y1.xls", sheet_name="2023-1", engine="xlrd")
b23_raw.columns = _flatten_cols(b23_raw.columns)
col23_name = _pick_col(b23_raw.columns.tolist(), ["\u05e9\u05dd \u05d9\u05d9\u05e9\u05d5\u05d1", "\u05d9\u05d9\u05e9\u05d5\u05d1"])
col23_pop = _pick_col(b23_raw.columns.tolist(), ["\u05e1\u05da \u05db\u05dc", "\u05ea\u05d5\u05e9\u05d1\u05d9\u05dd"])
col23_dis = _pick_col(b23_raw.columns.tolist(), ["\u05e0\u05db\u05d5\u05ea"])

b23_tmp = pd.DataFrame({
    "settlement_name_2023": b23_raw[col23_name],
    "population_2023": _to_number(b23_raw[col23_pop]).round(0).astype("Int64"),
    "disability_recipients_2023": _to_number(b23_raw[col23_dis]),
})
b23_tmp["name_norm"] = b23_tmp["settlement_name_2023"].apply(_normalize_name)
b23 = (
    b23_tmp.dropna(subset=["population_2023"])
    .groupby("name_norm", as_index=False)
    .agg({"population_2023": "sum", "disability_recipients_2023": "sum"})
)
b23["disability_rate_2023"] = (b23["disability_recipients_2023"] / b23["population_2023"]) * 100

# Merge 2024 + 2023
temporal_merged = b24.merge(b23, on="name_norm", how="left")

# 2025 data
d25_raw = pd.read_csv(DATA_DIR / "btl_disability_with_codes_2025.csv")
d25 = d25_raw.copy()
d25["settlement_code"] = pd.to_numeric(d25["settlement_code"], errors="coerce").astype("Int64")
d25["disability_recipients_2025"] = pd.to_numeric(d25["benefit_recipients"], errors="coerce")
d25["population_2025"] = pd.to_numeric(d25["residents"], errors="coerce").astype("Int64")
d25 = d25[["settlement_code", "disability_recipients_2025", "population_2025"]]

temporal_merged = temporal_merged.merge(d25, on="settlement_code", how="left")
temporal_merged["disability_rate_2025"] = (
    temporal_merged["disability_recipients_2025"] / temporal_merged["population_2025"]
) * 100

# Manual patches for missing 2025 frontline settlements
for patch_name, patch_pop, patch_recip in [
    ("\u05e9\u05dc\u05d5\u05de\u05d9", 7934, 354),
    ("\u05e2'\u05d2'\u05e8", 2926, 86),
]:
    mask = temporal_merged["settlement_name"].str.strip() == patch_name
    if mask.any():
        temporal_merged.loc[mask, "population_2025"] = patch_pop
        temporal_merged.loc[mask, "disability_recipients_2025"] = patch_recip
        temporal_merged.loc[mask, "disability_rate_2025"] = 100 * patch_recip / patch_pop

# Flag frontline settlements
temporal_merged["frontline"] = temporal_merged["settlement_code"].isin(frontline_codes)

# Build long-form temporal panel
years = [2023, 2024, 2025]
parts: list[pd.DataFrame] = []
for yr in years:
    tmp = temporal_merged[[
        "settlement_code", "settlement_name", "frontline",
        f"population_{yr}", f"disability_recipients_{yr}",
    ]].copy()
    tmp["year"] = yr
    tmp.rename(columns={
        f"population_{yr}": "population",
        f"disability_recipients_{yr}": "recipients",
    }, inplace=True)
    parts.append(tmp)

long_temporal = pd.concat(parts, ignore_index=True)
long_temporal["population"] = pd.to_numeric(long_temporal["population"], errors="coerce")
long_temporal["recipients"] = pd.to_numeric(long_temporal["recipients"], errors="coerce")
long_temporal = long_temporal.dropna(subset=["population", "recipients"])
long_temporal = long_temporal[long_temporal["population"] > 0]
long_temporal["group"] = np.where(long_temporal["frontline"], "Frontline", "Non-frontline")

# Weighted summaries
front_temporal = long_temporal[long_temporal["group"] == "Frontline"]
sum_front = _weighted_summary(front_temporal, "Frontline")
nf_temporal = long_temporal[long_temporal["group"] == "Non-frontline"]
nf_year_counts = nf_temporal.groupby("settlement_code")["year"].nunique()
complete_nf_codes = set(nf_year_counts[nf_year_counts == len(years)].index)
nf_balanced = nf_temporal[nf_temporal["settlement_code"].isin(complete_nf_codes)]
sum_nf = _weighted_summary(nf_balanced, "Non-frontline")

n_frontline_panel = int(sum_front["n_settlements"].iloc[0]) if len(sum_front) else 0
n_nonfrontline_panel = int(sum_nf["n_settlements"].iloc[0]) if len(sum_nf) else 0

# Temporal chart data
temporal_chart_data: dict = {"series": []}
for summary_df, series_name in [(sum_front, "Frontline (Swords of Iron)"), (sum_nf, "Non-Frontline")]:
    d = summary_df.sort_values("year")
    deltas = []
    rates = d["rate"].tolist()
    for i in range(len(rates)):
        if i == 0:
            deltas.append(None)
        else:
            deltas.append(round(rates[i] - rates[i - 1], 3))
    temporal_chart_data["series"].append({
        "name": series_name,
        "years": d["year"].tolist(),
        "rates": [round(float(r), 3) for r in rates],
        "deltas": deltas,
        "n_settlements": d["n_settlements"].tolist(),
        "total_pop": d["total_population"].tolist(),
        "total_recip": d["total_recipients"].tolist(),
    })

front_rates = sum_front.sort_values("year")["rate"].tolist() if len(sum_front) else []
nf_rates = sum_nf.sort_values("year")["rate"].tolist() if len(sum_nf) else []
front_rate_2023 = round(float(front_rates[0]), 2) if len(front_rates) > 0 else 0
front_rate_2025 = round(float(front_rates[-1]), 2) if len(front_rates) > 0 else 0
front_delta_total = round(front_rate_2025 - front_rate_2023, 2)
nf_rate_2023 = round(float(nf_rates[0]), 2) if len(nf_rates) > 0 else 0
nf_rate_2025 = round(float(nf_rates[-1]), 2) if len(nf_rates) > 0 else 0
nf_delta_total = round(nf_rate_2025 - nf_rate_2023, 2)

print(f"  Frontline: {n_frontline_panel} settlements, rate {front_rate_2023}% -> {front_rate_2025}% (delta={front_delta_total:+.2f})")
print(f"  Non-frontline: {n_nonfrontline_panel} settlements, rate {nf_rate_2023}% -> {nf_rate_2025}% (delta={nf_delta_total:+.2f})")

# ── Distance data (Q4) ───────────────────────────────────────────────────────

print("Loading distance data ...")
DIST_CSV = DATA_DIR / "data" / "processed" / "my_dataset_with_distances.csv"
df_dist = pd.read_csv(DIST_CSV)
df_dist["settlement_code"] = pd.to_numeric(df_dist["settlement_code"], errors="coerce")
for dc in ["dist_any_branch_km", "dist_central_branch_km"]:
    df_dist[dc] = pd.to_numeric(df_dist[dc], errors="coerce")

# Merge distances into main df
dist_map = df_dist.set_index("settlement_code")[["dist_any_branch_km", "dist_central_branch_km"]]
df["settlement_symbol_num"] = pd.to_numeric(df.get("settlement_symbol", pd.Series(dtype=float)), errors="coerce")
df = df.merge(dist_map, left_on="settlement_symbol_num", right_index=True, how="left")

# Distance scatter data
df_dist_plot = df.dropna(subset=["dist_any_branch_km", "general_disability_rate"]).copy()
dist_scatter_data = {
    "names": df_dist_plot["settlement_name"].tolist(),
    "disability_rate": [round(float(v), 2) for v in df_dist_plot["general_disability_rate"].tolist()],
    "dist_any": [round(float(v), 2) for v in df_dist_plot["dist_any_branch_km"].tolist()],
    "dist_central": [round(float(v), 2) for v in df_dist_plot["dist_central_branch_km"].tolist()],
}

# Trend line data (SES vs distance, z-scored)
ses_col = "socio_economic_index_score"
dist_col_name = "dist_any_branch_km"
df_trend = df[[ses_col, dist_col_name, "general_disability_rate"]].dropna().copy()

ses_mean, ses_std = df_trend[ses_col].mean(), df_trend[ses_col].std(ddof=0)
dist_mean, dist_std = df_trend[dist_col_name].mean(), df_trend[dist_col_name].std(ddof=0)
df_trend["ses_z"] = (df_trend[ses_col] - ses_mean) / ses_std
df_trend["dist_z"] = (df_trend[dist_col_name] - dist_mean) / dist_std

lr_ses = LinearRegression().fit(df_trend[["ses_z"]], df_trend["general_disability_rate"])
lr_dist = LinearRegression().fit(df_trend[["dist_z"]], df_trend["general_disability_rate"])

z_min = min(df_trend["ses_z"].min(), df_trend["dist_z"].min())
z_max = max(df_trend["ses_z"].max(), df_trend["dist_z"].max())
x_grid = np.linspace(z_min, z_max, 200)
y_ses = lr_ses.predict(x_grid.reshape(-1, 1))
y_dist = lr_dist.predict(x_grid.reshape(-1, 1))

trend_data = {
    "x_grid": [round(float(v), 3) for v in x_grid],
    "y_ses": [round(float(v), 3) for v in y_ses],
    "y_dist": [round(float(v), 3) for v in y_dist],
    "slope_ses": round(float(lr_ses.coef_[0]), 3),
    "slope_dist": round(float(lr_dist.coef_[0]), 3),
}
print(f"  Slope SES: {trend_data['slope_ses']}, Slope Distance: {trend_data['slope_dist']}")

n_total = len(df)
TARGET = "general_disability_rate"

# ── Model (15 features) ─────────────────────────────────────────────────────

print("Running model (15 features) ...")

FEATURES = [
    "socio_economic_index_score",
    "peripherality_index_score",
    "arab_population_percentage",
    "haredi_population_percentage",
    "unemployment_rate",
    "work_injury_victims_rate",
    "num_workers_2023",
    "age65_pct",
    "edu_higher_ed_entry_within_8y_pct",
    "edu_dropout_pct",
    "edu_attain_pct_academic_degree",
    "disabled_child_benefit_rate",
    "log_total_pop",
    "lat",
    "edu_attain_pct_no_info",
]

FEATURE_LABELS = {
    "socio_economic_index_score": "Socio-Economic Score (CBS)",
    "peripherality_index_score": "Peripherality Index",
    "arab_population_percentage": "Arab Population %",
    "haredi_population_percentage": "Haredi Population %",
    "unemployment_rate": "Unemployment Rate",
    "work_injury_victims_rate": "Work Injury Rate",
    "num_workers_2023": "Number of Workers",
    "age65_pct": "Population 65+ %",
    "edu_higher_ed_entry_within_8y_pct": "Higher Ed Entry Rate",
    "edu_dropout_pct": "School Dropout Rate",
    "edu_attain_pct_academic_degree": "Academic Degree %",
    "disabled_child_benefit_rate": "Child Disability Rate",
    "log_total_pop": "Settlement Size (log)",
    "lat": "Latitude (N-S position)",
    "edu_attain_pct_no_info": "Education Data Missing %",
}

df_reg = df[FEATURES + [TARGET, "settlement_name"]].dropna(subset=[TARGET]).copy()
for c in FEATURES:
    df_reg[c] = pd.to_numeric(df_reg[c], errors="coerce").astype(float)

X = df_reg[FEATURES].to_numpy(dtype=float, na_value=np.nan)
y = df_reg[TARGET].to_numpy(dtype=float)
n_reg = len(df_reg)
n_features = len(FEATURES)

# Spatial block CV
lat_new = pd.to_numeric(df.loc[df_reg.index, "lat"], errors="coerce")
geo_blocks_new = pd.cut(lat_new, bins=[-np.inf, 31.6, 32.2, 32.7, np.inf],
                         labels=["South", "Center", "North-Center", "North"])
group_ids = geo_blocks_new.cat.codes.values
cv = GroupKFold(n_splits=4)

rf_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestRegressor(
        n_estimators=600, min_samples_leaf=2,
        random_state=42, n_jobs=1,
    )),
])

rf_pred_oof = cross_val_predict(rf_pipe, X, y, cv=cv, groups=group_ids)
rf_scores = regression_metrics(y, rf_pred_oof)
print(f"  RF: R2 = {rf_scores['r2']}")

xgb_available = False
xgb_pred_oof = np.zeros(len(y), dtype=float)
xgb_scores = None

if HAS_XGBOOST:
    xgb_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", XGBRegressor(
            n_estimators=450, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42, n_jobs=1,
        )),
    ])
    try:
        xgb_pred_oof = cross_val_predict(xgb_pipe, X, y, cv=cv, groups=group_ids)
        xgb_scores = regression_metrics(y, xgb_pred_oof)
        xgb_available = True
        print(f"  XGB: R2 = {xgb_scores['r2']}")
    except Exception:
        pass

# TabPFN
tabpfn_available = False
tabpfn_pred_oof = np.zeros(len(y), dtype=float)
tabpfn_scores = None

if HAS_TABPFN:
    try:
        print("Running TabPFN ...")
        tabpfn_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("tabpfn", TabPFNRegressor.create_default_for_version("v2")),
        ])
        tabpfn_pred_oof = cross_val_predict(tabpfn_pipe, X, y, cv=cv, groups=group_ids)
        tabpfn_scores = regression_metrics(y, tabpfn_pred_oof)
        tabpfn_available = True
        print(f"  TabPFN: R2 = {tabpfn_scores['r2']}")
    except Exception as e:
        print(f"  TabPFN unavailable: {e}")

# Best prediction
if tabpfn_available and xgb_available:
    mean_pred_oof = (rf_pred_oof + xgb_pred_oof + tabpfn_pred_oof) / 3.0
elif tabpfn_available:
    mean_pred_oof = (rf_pred_oof + tabpfn_pred_oof) / 2.0
elif xgb_available:
    mean_pred_oof = (rf_pred_oof + xgb_pred_oof) / 2.0
else:
    mean_pred_oof = rf_pred_oof

# Feature importance (average RF + XGB if available)
rf_pipe.fit(X, y)
rf_importances = rf_pipe.named_steps["rf"].feature_importances_.astype(float)

if xgb_available:
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    xgb_pipe.named_steps["xgb"].fit(X_imp, y)
    xgb_importances = xgb_pipe.named_steps["xgb"].feature_importances_.astype(float)
    avg_importances = (rf_importances + xgb_importances) / 2.0
else:
    avg_importances = rf_importances

# Build importance data for chart (sorted descending)
imp_order = np.argsort(avg_importances)[::-1]
importance_data = {
    "labels": [FEATURE_LABELS.get(FEATURES[i], FEATURES[i]) for i in imp_order],
    "values": [round(float(avg_importances[i]), 4) for i in imp_order],
}

# ── OLS Linear Regression ────────────────────────────────────────────────────

print("Running OLS linear regression (8 features) ...")

salary_col = "average_monthly_salary_2023_imputed" if "average_monthly_salary_2023_imputed" in df.columns else "average_monthly_salary_2023"
df_ols = df.loc[df_reg.index].copy()
df_ols["log_salary"] = np.log(pd.to_numeric(df_ols[salary_col], errors="coerce"))
df_ols["pct_65_plus"] = (pd.to_numeric(df_ols["population_65_plus"], errors="coerce")
                          / pd.to_numeric(df_ols["total_population"], errors="coerce")) * 100
pop_working = (pd.to_numeric(df_ols["population_18_64"], errors="coerce")
               + pd.to_numeric(df_ols["population_65_plus"], errors="coerce"))
df_ols["labor_participation"] = (pd.to_numeric(df_ols["num_workers_2023"], errors="coerce")
                                  / pop_working * 100).clip(upper=100.0)
pop_0_17 = pd.to_numeric(df_ols["population_0_17"], errors="coerce")
pop_18_64 = pd.to_numeric(df_ols["population_18_64"], errors="coerce")
pop_65 = pd.to_numeric(df_ols["population_65_plus"], errors="coerce")
df_ols["dependency_ratio"] = ((pop_0_17 + pop_65) / pop_18_64) * 100

OLS_FEATURES = [
    "log_salary", "peripherality_index_score", "arab_population_percentage",
    "haredi_population_percentage", "income_support_rate", "pct_65_plus",
    "labor_participation", "dependency_ratio",
]
OLS_LABELS = {
    "log_salary": "Log Salary",
    "peripherality_index_score": "Peripherality Index",
    "arab_population_percentage": "Arab Population %",
    "haredi_population_percentage": "Haredi Population %",
    "income_support_rate": "Income Support Rate",
    "pct_65_plus": "Population 65+ %",
    "labor_participation": "Labor Participation",
    "dependency_ratio": "Dependency Ratio",
}

for c in OLS_FEATURES:
    df_ols[c] = pd.to_numeric(df_ols.get(c, pd.Series(dtype=float)), errors="coerce").astype(float)

df_ols_clean = df_ols.dropna(subset=OLS_FEATURES + [TARGET])
X_ols_raw = df_ols_clean[OLS_FEATURES].values
y_ols = df_ols_clean[TARGET].values

scaler = StandardScaler()
X_ols_scaled = scaler.fit_transform(X_ols_raw)

ols_model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
ols_cv_scores = cross_val_score(ols_model, X_ols_scaled, y_ols, cv=kf, scoring="r2")
ols_cv_r2 = round(float(ols_cv_scores.mean()), 3)
ols_cv_std = round(float(ols_cv_scores.std()), 3)

# Fit on full data for coefficients and residuals
ols_model.fit(X_ols_scaled, y_ols)
ols_r2_full = round(float(r2_score(y_ols, ols_model.predict(X_ols_scaled))), 3)
ols_coefficients = ols_model.coef_.astype(float)

# Coefficient chart data (sorted by absolute magnitude)
ols_abs_order = np.argsort(np.abs(ols_coefficients))[::-1]
ols_coeff_data = {
    "labels": [OLS_LABELS.get(OLS_FEATURES[i], OLS_FEATURES[i]) for i in ols_abs_order],
    "values": [round(float(ols_coefficients[i]), 4) for i in ols_abs_order],
    "colors": ["#C44E52" if ols_coefficients[i] > 0 else "#4C72B0" for i in ols_abs_order],
}

print(f"  OLS: CV R2 = {ols_cv_r2} +/- {ols_cv_std}, Full R2 = {ols_r2_full}")

# Anomaly detection (residual-based)
y_ols_pred = ols_model.predict(X_ols_scaled)
ols_residuals = y_ols - y_ols_pred
ols_threshold = 1.5 * float(np.std(ols_residuals))

ols_names = df_ols_clean["settlement_name"].tolist() if "settlement_name" in df_ols_clean.columns else []
ols_total_pop = pd.to_numeric(df_ols_clean.get("total_population", pd.Series(dtype=float)), errors="coerce").values

anomaly_types = []
for r in ols_residuals:
    if r < -ols_threshold:
        anomaly_types.append("Underutilization")
    elif r > ols_threshold:
        anomaly_types.append("Dependency")
    else:
        anomaly_types.append("Normal")

ols_anomaly_data = {}
for atype in ["Normal", "Underutilization", "Dependency"]:
    mask_at = [t == atype for t in anomaly_types]
    idx = [i for i, m in enumerate(mask_at) if m]
    ols_anomaly_data[atype] = {
        "predicted": [round(float(y_ols_pred[i]), 2) for i in idx],
        "actual": [round(float(y_ols[i]), 2) for i in idx],
        "names": [ols_names[i] for i in idx] if ols_names else [],
        "residual": [round(float(ols_residuals[i]), 2) for i in idx],
        "population": [int(ols_total_pop[i]) if not np.isnan(ols_total_pop[i]) else 0 for i in idx],
    }

n_ols_under = sum(1 for t in anomaly_types if t == "Underutilization")
n_ols_dependency = sum(1 for t in anomaly_types if t == "Dependency")

# Top anomalies sorted by residual (for the table)
top_under_table = sorted(
    [(ols_names[i], round(float(y_ols[i]), 2), round(float(y_ols_pred[i]), 2),
      round(float(ols_residuals[i]), 2), int(ols_total_pop[i]) if not np.isnan(ols_total_pop[i]) else 0)
     for i in range(len(ols_residuals)) if anomaly_types[i] == "Underutilization"],
    key=lambda x: x[3]
)[:10]

ols_min_val = float(min(min(y_ols), min(y_ols_pred)))
ols_max_val = float(max(max(y_ols), max(y_ols_pred)))

print(f"  Anomaly detection: {n_ols_under} underutilization, {n_ols_dependency} dependency (threshold +/-{ols_threshold:.2f})")

# ── Sector R2 ───────────────────────────────────────────────────────────────

print("Computing sector R2 ...")

arab_pct = df_reg["arab_population_percentage"].values
haredi_pct = df_reg["haredi_population_percentage"].values
arab_mask = arab_pct > 50
haredi_mask = haredi_pct > 20
secular_mask = (~arab_mask) & (~haredi_mask)

sector_rows = []
for sname, smask in [
    ("Arab >50%", arab_mask), ("Haredi >20%", haredi_mask),
    ("Secular", secular_mask), ("All 278", np.ones(len(y), dtype=bool)),
]:
    n = int(smask.sum())
    row = {"sector": sname, "n": n}
    row["rf_r2"] = round(r2_score(y[smask], rf_pred_oof[smask]), 3) if n >= 5 else None
    row["xgb_r2"] = round(r2_score(y[smask], xgb_pred_oof[smask]), 3) if xgb_available and n >= 5 else None
    row["tabpfn_r2"] = round(r2_score(y[smask], tabpfn_pred_oof[smask]), 3) if tabpfn_available and n >= 5 else None
    sector_rows.append(row)

arab_r2_vals = [v for row in sector_rows if row["sector"] == "Arab >50%"
                for k, v in row.items() if k.endswith("_r2") and v is not None]
arab_best_r2 = max(arab_r2_vals) if arab_r2_vals else 0.0
arab_unexplained_pct = round((1 - arab_best_r2) * 100)

overall_r2_vals = [v for row in sector_rows if row["sector"] == "All 278"
                   for k, v in row.items() if k.endswith("_r2") and v is not None]
best_overall_r2 = max(overall_r2_vals) if overall_r2_vals else 0.0

secular_r2_vals = [v for row in sector_rows if row["sector"] == "Secular"
                   for k, v in row.items() if k.endswith("_r2") and v is not None]
best_secular_r2 = max(secular_r2_vals) if secular_r2_vals else 0.0

# Sector chart data for grouped bar
arab_row = next(r for r in sector_rows if r["sector"] == "Arab >50%")
all_row = next(r for r in sector_rows if r["sector"] == "All 278")
secular_row = next(r for r in sector_rows if r["sector"] == "Secular")

arab_chart_data = {"models": [], "arab_r2": [], "overall_r2": [], "secular_r2": []}
arab_chart_data["models"].append("RandomForest")
arab_chart_data["arab_r2"].append(arab_row["rf_r2"])
arab_chart_data["overall_r2"].append(all_row["rf_r2"])
arab_chart_data["secular_r2"].append(secular_row["rf_r2"])
if xgb_available:
    arab_chart_data["models"].append("XGBoost")
    arab_chart_data["arab_r2"].append(arab_row["xgb_r2"])
    arab_chart_data["overall_r2"].append(all_row["xgb_r2"])
    arab_chart_data["secular_r2"].append(secular_row["xgb_r2"])
if tabpfn_available:
    arab_chart_data["models"].append("TabPFN v2")
    arab_chart_data["arab_r2"].append(arab_row["tabpfn_r2"])
    arab_chart_data["overall_r2"].append(all_row["tabpfn_r2"])
    arab_chart_data["secular_r2"].append(secular_row["tabpfn_r2"])

best_model = "TabPFN v2" if tabpfn_available else ("XGBoost" if xgb_available else "RandomForest")
best_r2_val = (tabpfn_scores or xgb_scores or rf_scores)["r2"]

# ── Gap analysis ─────────────────────────────────────────────────────────────

print("Computing gap analysis ...")

extra_cols = ["settlement_name", "socio_economic_index_score",
              "arab_population_percentage", "haredi_population_percentage",
              "age65_pct"]
extra_cols = [c for c in extra_cols if c in df.columns]

df_gap = df.loc[df_reg.index, extra_cols].copy()
df_gap["actual_rate"] = y
df_gap["rf_pred_rate"] = rf_pred_oof
if xgb_available:
    df_gap["xgb_pred_rate"] = xgb_pred_oof
df_gap["mean_pred_rate"] = mean_pred_oof
df_gap["gap_pp"] = df_gap["mean_pred_rate"] - df_gap["actual_rate"]

actual_high_threshold = float(np.quantile(y, 0.75))
expected_high_threshold = float(np.quantile(mean_pred_oof, 0.75))


def _segment(row: pd.Series) -> str:
    if row["mean_pred_rate"] >= expected_high_threshold and row["actual_rate"] < actual_high_threshold:
        return "Potential Under-utilization"
    if row["mean_pred_rate"] < expected_high_threshold and row["actual_rate"] >= actual_high_threshold:
        return "Hidden Burden"
    if row["mean_pred_rate"] >= expected_high_threshold and row["actual_rate"] >= actual_high_threshold:
        return "Both High"
    return "Both Lower"


df_gap["segment"] = df_gap.apply(_segment, axis=1)

segment_order = ["Both Lower", "Both High", "Potential Under-utilization", "Hidden Burden"]
gap_scatter_data: dict[str, dict] = {}
for seg in segment_order:
    sub = df_gap[df_gap["segment"] == seg]
    gap_scatter_data[seg] = {
        "x": sub["actual_rate"].round(2).tolist(),
        "y": sub["mean_pred_rate"].round(2).tolist(),
        "names": sub["settlement_name"].tolist() if "settlement_name" in sub.columns else [],
        "gap": sub["gap_pp"].round(2).tolist(),
    }

under_candidates = df_gap[df_gap["segment"] == "Potential Under-utilization"].sort_values(
    "gap_pp", ascending=False
)
hidden_candidates = df_gap[df_gap["segment"] == "Hidden Burden"].sort_values(
    "actual_rate", ascending=False
)
n_under = len(under_candidates)
n_hidden = len(hidden_candidates)

# Arab deep-dive
arab_pct_col = pd.to_numeric(
    df_gap.get("arab_population_percentage", pd.Series(dtype=float)), errors="coerce"
).fillna(0)
arab_under = under_candidates[arab_pct_col.reindex(under_candidates.index).fillna(0) > 50]
n_arab = int(arab_mask.sum())
n_arab_under = len(arab_under)
under_arab_pct = round(len(arab_under) / n_under * 100) if n_under > 0 else 0

# ── Intergenerational analysis (Q2) ─────────────────────────────────────────

print("Computing intergenerational analysis ...")

child_rate_col = pd.to_numeric(df.loc[df_reg.index, "disabled_child_benefit_rate"], errors="coerce")
adult_rate_col = pd.to_numeric(df.loc[df_reg.index, "general_disability_rate"], errors="coerce")
periph_cluster = pd.to_numeric(df.loc[df_reg.index, "peripherality_index_cluster"], errors="coerce")

inter_valid = child_rate_col.notna() & adult_rate_col.notna()
rho_overall, pval_overall = spearmanr(
    child_rate_col[inter_valid], adult_rate_col[inter_valid]
)
rho_overall = round(rho_overall, 3)

periph_periphery = periph_cluster.isin([1, 2, 3])
periph_center = periph_cluster.isin([8, 9, 10])

mask_periph = inter_valid & periph_periphery
rho_periph, _ = spearmanr(child_rate_col[mask_periph], adult_rate_col[mask_periph])
rho_periph = round(rho_periph, 3)
n_periph = int(mask_periph.sum())

mask_center = inter_valid & periph_center
rho_center, _ = spearmanr(child_rate_col[mask_center], adult_rate_col[mask_center])
rho_center = round(rho_center, 3)
n_center = int(mask_center.sum())

n_inter = int(inter_valid.sum())

# Scatter data for Chart.js
inter_scatter_data = {"groups": []}
for group_label, group_mask, color in [
    ("Periphery (1-3)", periph_periphery & inter_valid, "#DD8452"),
    ("Middle (4-7)", (~periph_periphery & ~periph_center) & inter_valid, "#8172B3"),
    ("Center (8-10)", periph_center & inter_valid, "#4C72B0"),
]:
    sub_child = child_rate_col[group_mask].round(2).tolist()
    sub_adult = adult_rate_col[group_mask].round(2).tolist()
    sub_names = df.loc[df_reg.index[group_mask], "settlement_name"].tolist() if "settlement_name" in df.columns else []
    inter_scatter_data["groups"].append({
        "name": group_label,
        "child_rate": sub_child,
        "adult_rate": sub_adult,
        "names": sub_names,
        "color": color,
    })

print(f"  Overall rho={rho_overall}, Periphery rho={rho_periph} (N={n_periph}), Center rho={rho_center} (N={n_center})")


# ── HTML generation ──────────────────────────────────────────────────────────

print("Generating HTML ...")

# Build top-10 underutilization table rows for Section 3
table_rows_html = ""
for i, (name, actual, predicted, residual, pop) in enumerate(top_under_table):
    table_rows_html += f"""<tr>
      <td>{i + 1}</td>
      <td>{html_lib.escape(str(name))}</td>
      <td>{actual:.2f}%</td>
      <td>{predicted:.2f}%</td>
      <td style="color:var(--positive);font-weight:600;">{residual:+.2f}</td>
      <td>{pop:,}</td>
    </tr>"""

# ── Assemble the full HTML ──────────────────────────────────────────────────

html_output = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Disability Under-Utilization: Where Need Meets Silence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.5.1" integrity="sha384-jb8JQMbMoBUzgWatfe6COACi2ljcDdZQ2OxczGA3bGNeWe+6DChMTBJemed7ZnvJ" crossorigin="anonymous"></script>
  <style>
    :root {{
      --bg-primary: #f8f9fa;
      --bg-card: #ffffff;
      --bg-header: #1a1a2e;
      --text-primary: #212529;
      --text-secondary: #6c757d;
      --text-on-dark: #ffffff;
      --color-1: #4C72B0;
      --color-2: #DD8452;
      --color-3: #55A868;
      --color-4: #C44E52;
      --color-5: #8172B3;
      --color-6: #937860;
      --positive: #28a745;
      --negative: #dc3545;
      --gap: 16px;
      --radius: 8px;
    }}

    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
      font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      line-height: 1.6;
      -webkit-font-smoothing: antialiased;
    }}

    /* ── Navigation ── */
    .top-nav {{
      position: sticky;
      top: 0;
      z-index: 100;
      background: var(--bg-header);
      color: var(--text-on-dark);
      display: flex;
      align-items: center;
      padding: 0 24px;
      height: 56px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }}
    .top-nav .brand {{
      font-weight: 700;
      font-size: 1rem;
      margin-right: 32px;
      white-space: nowrap;
    }}
    .top-nav .nav-links {{
      display: flex;
      gap: 4px;
      overflow-x: auto;
      scrollbar-width: none;
    }}
    .top-nav .nav-links::-webkit-scrollbar {{ display: none; }}
    .top-nav .nav-links a {{
      color: rgba(255,255,255,0.7);
      text-decoration: none;
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 500;
      white-space: nowrap;
      transition: all 0.2s;
    }}
    .top-nav .nav-links a:hover {{
      color: #fff;
      background: rgba(255,255,255,0.1);
    }}
    .top-nav .nav-links a.active {{
      color: #fff;
      background: var(--color-1);
    }}

    /* ── Layout ── */
    .dashboard {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px var(--gap);
    }}

    .section {{
      margin-bottom: 48px;
      scroll-margin-top: 72px;
    }}

    .section-title {{
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--text-primary);
      margin-bottom: 4px;
    }}

    .section-subtitle {{
      font-size: 0.95rem;
      color: var(--text-secondary);
      margin-bottom: 20px;
    }}

    /* ── KPI Cards ── */
    .kpi-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: var(--gap);
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--bg-card);
      border-radius: var(--radius);
      padding: 20px 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      border: 1px solid #e9ecef;
      text-align: center;
    }}
    .kpi-card .kpi-value {{
      font-size: 2.2rem;
      font-weight: 700;
      color: var(--color-1);
      line-height: 1.2;
    }}
    .kpi-card .kpi-value.accent {{ color: var(--color-2); }}
    .kpi-card .kpi-value.positive {{ color: var(--positive); }}
    .kpi-card .kpi-value.negative {{ color: var(--negative); }}
    .kpi-card .kpi-label {{
      font-size: 0.82rem;
      color: var(--text-secondary);
      margin-top: 6px;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    /* ── Cards grid ── */
    .card-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: var(--gap);
    }}
    .card-grid.three {{ grid-template-columns: repeat(3, 1fr); }}
    .card-grid.full {{ grid-template-columns: 1fr; }}

    .card {{
      background: var(--bg-card);
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      border: 1px solid #e9ecef;
    }}
    .card h3 {{
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 12px;
      color: var(--text-primary);
    }}
    .card p {{
      font-size: 0.9rem;
      color: var(--text-secondary);
      line-height: 1.7;
    }}
    .card ul {{
      padding-left: 18px;
      margin-top: 8px;
    }}
    .card li {{
      font-size: 0.9rem;
      color: var(--text-secondary);
      margin-bottom: 4px;
    }}
    .card strong {{ color: var(--text-primary); }}

    /* ── Chart wrapper ── */
    .chart-card {{
      background: var(--bg-card);
      border-radius: var(--radius);
      padding: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      border: 1px solid #e9ecef;
      margin-bottom: var(--gap);
    }}
    .chart-card .chart-title {{
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-primary);
      margin-bottom: 12px;
    }}
    .chart-card .chart-wrap {{
      position: relative;
      width: 100%;
      height: 340px;
    }}
    .chart-card .chart-wrap.tall {{
      height: 420px;
    }}
    .chart-card canvas {{
      width: 100% !important;
      max-height: 100%;
    }}

    /* ── Data Table ── */
    .data-table-wrap {{
      background: var(--bg-card);
      border-radius: var(--radius);
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      border: 1px solid #e9ecef;
      overflow-x: auto;
      margin-bottom: var(--gap);
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }}
    .data-table th {{
      background: #f1f3f5;
      padding: 10px 14px;
      text-align: left;
      font-weight: 600;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.03em;
      font-size: 0.75rem;
      border-bottom: 2px solid #dee2e6;
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }}
    .data-table th:hover {{ background: #e9ecef; }}
    .data-table th .sort-arrow {{ margin-left: 4px; opacity: 0.4; }}
    .data-table th.sorted .sort-arrow {{ opacity: 1; }}
    .data-table td {{
      padding: 10px 14px;
      border-bottom: 1px solid #f1f3f5;
      color: var(--text-primary);
    }}
    .data-table tbody tr:hover {{
      background: #f8f9fa;
    }}

    /* ── Big Number ── */
    .big-number-card {{
      background: var(--bg-card);
      border-radius: var(--radius);
      padding: 32px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      border: 1px solid #e9ecef;
      text-align: center;
      margin-bottom: var(--gap);
    }}
    .big-number-card .big-val {{
      font-size: 3.5rem;
      font-weight: 700;
      color: var(--negative);
      line-height: 1.1;
    }}
    .big-number-card .big-desc {{
      font-size: 1rem;
      color: var(--text-secondary);
      margin-top: 8px;
      max-width: 500px;
      margin-left: auto;
      margin-right: auto;
    }}

    /* ── Insight callout ── */
    .insight-callout {{
      background: #e8f4fd;
      border-left: 4px solid var(--color-1);
      border-radius: 0 var(--radius) var(--radius) 0;
      padding: 16px 20px;
      margin-bottom: var(--gap);
      font-size: 0.9rem;
      color: #1a5276;
    }}
    .insight-callout strong {{ color: #154360; }}

    /* ── Rec cards ── */
    .rec-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: var(--gap);
    }}
    .rec-card {{
      background: var(--bg-card);
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      border: 1px solid #e9ecef;
      border-top: 3px solid var(--color-1);
    }}
    .rec-card .rec-num {{
      display: inline-block;
      background: var(--color-1);
      color: #fff;
      width: 28px;
      height: 28px;
      border-radius: 50%;
      text-align: center;
      line-height: 28px;
      font-weight: 700;
      font-size: 0.85rem;
      margin-bottom: 10px;
    }}
    .rec-card h4 {{
      font-size: 0.95rem;
      font-weight: 600;
      margin-bottom: 8px;
    }}
    .rec-card p {{
      font-size: 0.85rem;
      color: var(--text-secondary);
      line-height: 1.6;
    }}

    /* ── Header banner ── */
    .header-banner {{
      background: var(--bg-header);
      color: var(--text-on-dark);
      padding: 48px 24px 40px;
      text-align: center;
    }}
    .header-banner h1 {{
      font-size: 2rem;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .header-banner .subtitle {{
      font-size: 1.05rem;
      color: rgba(255,255,255,0.7);
      margin-bottom: 16px;
    }}
    .header-banner .meta {{
      font-size: 0.82rem;
      color: rgba(255,255,255,0.5);
    }}

    /* ── Responsive ── */
    @media (max-width: 768px) {{
      .card-grid {{ grid-template-columns: 1fr; }}
      .card-grid.three {{ grid-template-columns: 1fr; }}
      .kpi-row {{ grid-template-columns: 1fr 1fr; }}
      .top-nav .brand {{ display: none; }}
      .header-banner h1 {{ font-size: 1.4rem; }}
      .big-number-card .big-val {{ font-size: 2.5rem; }}
      .rec-grid {{ grid-template-columns: 1fr; }}
      .chart-card .chart-wrap {{ height: 280px; }}
      .chart-card .chart-wrap.tall {{ height: 320px; }}
    }}

    /* ── Print ── */
    @media print {{
      .top-nav {{ display: none; }}
      body {{ background: #fff; }}
      .section {{ break-inside: avoid; page-break-inside: avoid; }}
      .chart-card {{ break-inside: avoid; page-break-inside: avoid; }}
      .chart-card, .card, .kpi-card {{ box-shadow: none; border: 1px solid #ccc; }}
    }}
  </style>
</head>
<body>

<!-- Navigation -->
<nav class="top-nav">
  <div class="brand">Where Need Meets Silence</div>
  <div class="nav-links">
    <a href="#overview" class="active">Overview</a>
    <a href="#drivers">Drivers</a>
    <a href="#underutil">Q1: Under-Utilization</a>
    <a href="#intergen">Q2: Intergenerational</a>
    <a href="#arabwall">The Wall</a>
    <a href="#conflict">Q3: Swords of Iron</a>
    <a href="#distance">Q4: Distance</a>
    <a href="#evidence">Evidence</a>
    <a href="#recommendations">Recommendations</a>
  </div>
</nav>

<!-- Header Banner -->
<header class="header-banner">
  <h1>Disability Under-Utilization: Where Need Meets Silence</h1>
  <p class="subtitle">Machine Learning Reveals Hidden Access Barriers Across 278 Israeli Settlements</p>
  <p class="meta">TovTech Research Group &nbsp;|&nbsp; Data: National Insurance Institute, Dec 2024 &nbsp;|&nbsp; EXPERIMENTAL DASHBOARD</p>
</header>

<div class="dashboard">

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 1: Overview + KPI Cards -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="overview">
  <h2 class="section-title">Overview</h2>
  <p class="section-subtitle">Key metrics from the analysis of Israeli settlement-level disability benefit data</p>

  <div class="kpi-row">
    <div class="kpi-card">
      <div class="kpi-value">{n_reg}</div>
      <div class="kpi-label">Settlements Analyzed</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value accent">{n_features}</div>
      <div class="kpi-label">Socio-Economic Features</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value positive">{best_r2_val}</div>
      <div class="kpi-label">Best Model R&sup2;</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value negative">{n_under}</div>
      <div class="kpi-label">Flagged Under-Utilization</div>
    </div>
  </div>

  <div class="insight-callout">
    <strong>Four research questions, one conclusion:</strong> disability benefit under-utilization
    is concentrated, intergenerational, structural, and invisible to standard data. Our models
    explain ~{round(best_r2_val * 100)}% of the variation overall but hit a wall for Arab
    settlements (R&sup2; = {arab_best_r2:.2f}).
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 2: What Drives Disability Rates? -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="drivers">
  <h2 class="section-title">Income Support Rate Is the Strongest Driver of Disability</h2>
  <p class="section-subtitle">Two complementary views: OLS reveals direction, tree models reveal importance</p>

  <div class="card-grid">
    <div class="chart-card">
      <div class="chart-title">OLS Standardized Coefficients &mdash; Red Increases, Blue Decreases Disability</div>
      <div class="chart-wrap"><canvas id="chart_ols_coeff"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-title">RF/XGB Feature Importance &mdash; Which Variables Matter Most</div>
      <div class="chart-wrap"><canvas id="chart_importance"></canvas></div>
    </div>
  </div>

  <div class="insight-callout">
    <strong>Why two charts?</strong> The OLS chart (left) shows <em>direction</em>:
    red bars increase disability rates, blue bars decrease them. Income support rate
    is the strongest positive driver; salary is the strongest negative driver.
    The tree model chart (right) shows <em>magnitude</em> of importance regardless
    of direction. Both methods agree on the same key factors.
    OLS R&sup2; = {ols_cv_r2} (5-fold CV). Best tree R&sup2; = {best_r2_val}.
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 3: Under-Utilization Pockets (Q1) -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="underutil">
  <h2 class="section-title">{n_under} Settlements Claim Far Less Than Expected</h2>
  <p class="section-subtitle">Q1: Ensemble model (RF + XGB + TabPFN) flags potential under-utilization, OLS confirms with residual-based anomalies</p>

  <div class="kpi-row">
    <div class="kpi-card">
      <div class="kpi-value positive">{n_under}</div>
      <div class="kpi-label">Under-Utilization (Ensemble)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value negative">{n_hidden}</div>
      <div class="kpi-label">Hidden Burden (Ensemble)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value">{under_arab_pct}%</div>
      <div class="kpi-label">Flagged Are Arab-Majority</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value accent">{best_r2_val}</div>
      <div class="kpi-label">Best Model R&sup2;</div>
    </div>
  </div>

  <div class="chart-card">
    <div class="chart-title">Ensemble Model: Expected vs Observed Disability Rate &mdash; 4 Segments (RF + XGB + TabPFN)</div>
    <div class="chart-wrap tall"><canvas id="chart_gap_scatter"></canvas></div>
  </div>

  <div class="insight-callout">
    <strong>Reading the chart above:</strong>
    <strong style="color:#F59E0B;">Gold diamonds</strong> = {n_under} settlements where observed claiming is far below expected (potential under-utilization).
    <strong style="color:#F43F5E;">Red crosses</strong> = {n_hidden} settlements where observed claiming exceeds expectations (hidden burden).
    The diagonal line = perfect prediction. This chart uses the average of our 3 tree models.
  </div>

  <div class="chart-card">
    <div class="chart-title">OLS Independent Validation: Actual vs Predicted (threshold &plusmn;1.5&sigma;) &mdash; {n_ols_under} under-utilization, {n_ols_dependency} dependency</div>
    <div class="chart-wrap tall"><canvas id="chart_anomaly_scatter"></canvas></div>
  </div>

  <h3 style="font-size:1rem;font-weight:600;margin:16px 0 8px;">Top 10 Under-Utilization Settlements</h3>
  <div class="data-table-wrap">
    <table class="data-table" id="anomaly_table">
      <thead>
        <tr>
          <th>#</th>
          <th onclick="sortTable('anomaly_table', 1)">Settlement <span class="sort-arrow">&#9650;&#9660;</span></th>
          <th onclick="sortTable('anomaly_table', 2)">Actual Rate <span class="sort-arrow">&#9650;&#9660;</span></th>
          <th onclick="sortTable('anomaly_table', 3)">Predicted Rate <span class="sort-arrow">&#9650;&#9660;</span></th>
          <th onclick="sortTable('anomaly_table', 4)">Residual <span class="sort-arrow">&#9650;&#9660;</span></th>
          <th onclick="sortTable('anomaly_table', 5)">Population <span class="sort-arrow">&#9650;&#9660;</span></th>
        </tr>
      </thead>
      <tbody>
        {table_rows_html}
      </tbody>
    </table>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 4: Intergenerational Trap (Q2) -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="intergen">
  <h2 class="section-title">The Pattern Persists Even in Affluent Areas</h2>
  <p class="section-subtitle">Q2: Child disability benefit rate correlates with adult disability &mdash; &rho; = {rho_overall}</p>

  <div class="kpi-row">
    <div class="kpi-card">
      <div class="kpi-value">{rho_overall}</div>
      <div class="kpi-label">Overall Spearman &rho;</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value accent">{rho_periph}</div>
      <div class="kpi-label">Periphery &rho; (N={n_periph})</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value" style="color:var(--color-1);">{rho_center}</div>
      <div class="kpi-label">Center &rho; (N={n_center})</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value">{n_inter}</div>
      <div class="kpi-label">Settlements with Both Rates</div>
    </div>
  </div>

  <div class="chart-card">
    <div class="chart-title">Child Disability Rate vs Adult Disability Rate by Peripherality</div>
    <div class="chart-wrap tall"><canvas id="chart_intergenerational"></canvas></div>
  </div>

  <div class="card-grid">
    <div class="card">
      <h3>What This Means</h3>
      <p>Settlements where more children receive disability benefits also have higher
      adult disability rates. This suggests a <strong>cycle of vulnerability</strong>:
      conditions that lead to childhood disability persist into adulthood.</p>
    </div>
    <div class="card">
      <h3>Stronger in the Center</h3>
      <p>Paradoxically, the center shows a <strong>stronger</strong> correlation (&rho; = {rho_center})
      than the periphery (&rho; = {rho_periph}). Even in affluent areas, intergenerational
      disability patterns persist, suggesting structural rather than purely economic causes.</p>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 5: The Wall — Q1 + Q2 Converge on the Arab Sector -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="arabwall">
  <h2 class="section-title">The Wall: Q1 + Q2 Converge on the Arab Sector</h2>
  <p class="section-subtitle">Both under-utilization (Q1) and the intergenerational trap (Q2) point to the same place &mdash; {n_arab} Arab-majority settlements</p>

  <div class="big-number-card">
    <div class="big-val">~{arab_unexplained_pct}%</div>
    <div class="big-desc">of the variance in Arab disability claiming <strong>cannot be explained</strong>
    by any combination of {n_features} socio-economic indicators</div>
  </div>

  <div class="chart-card">
    <div class="chart-title">Model R&sup2; by Population Sector &mdash; Best model R&sup2; = {arab_best_r2:.3f} for Arab vs {fmt_num(best_secular_r2, 3)} for Secular</div>
    <div class="chart-wrap"><canvas id="chart_arab_r2"></canvas></div>
  </div>

  <div class="card-grid">
    <div class="card">
      <h3>Q1 Signal: Under-Utilization Is Concentrated Here</h3>
      <p>Of the {n_under} flagged settlements, <strong>{under_arab_pct}%</strong> are
      Arab-majority. The model <em>predicts</em> high disability rates, but observed
      claiming is far <em>below</em> prediction. Standard factors (income, education,
      demographics) explain R&sup2; = {fmt_num(best_secular_r2, 3)} for secular settlements
      but almost nothing for Arab ones.</p>
    </div>
    <div class="card">
      <h3>Q2 Signal: The Intergenerational Pattern Is Present</h3>
      <p>The child-adult disability correlation exists across all sectors,
      but in Arab communities, the <strong>model residual</strong> is largest &mdash;
      meaning the gap between predicted and actual is not random.
      Something <strong>invisible to official statistics</strong> drives
      the gap, and Q2 shows this pattern starts in childhood.
      This convergence points to <strong>access barriers</strong>, not lower need.</p>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 6: Swords of Iron (Q3) -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="conflict">
  <h2 class="section-title">The Rise Is Structural, Not Crisis-Driven</h2>
  <p class="section-subtitle">Q3: Disability rates 2023-2025 &mdash; frontline vs non-frontline settlements</p>

  <div class="kpi-row">
    <div class="kpi-card">
      <div class="kpi-value negative">{front_rate_2023}%</div>
      <div class="kpi-label">Frontline 2023</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value negative">{front_rate_2025}%</div>
      <div class="kpi-label">Frontline 2025</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value">{nf_rate_2023}%</div>
      <div class="kpi-label">Non-Frontline 2023</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value">{nf_rate_2025}%</div>
      <div class="kpi-label">Non-Frontline 2025</div>
    </div>
  </div>

  <div class="chart-card">
    <div class="chart-title">Disability Benefit Rate: Frontline ({n_frontline_panel} settlements) vs Non-Frontline ({n_nonfrontline_panel} settlements)</div>
    <div class="chart-wrap"><canvas id="chart_temporal"></canvas></div>
  </div>

  <div class="insight-callout">
    <strong>Key finding:</strong> Disability rates rose <strong>everywhere</strong>, not just in
    frontline zones. Frontline: {front_rate_2023}% &rarr; {front_rate_2025}%
    (&Delta; = {front_delta_total:+.2f} pp). Non-frontline: {nf_rate_2023}% &rarr; {nf_rate_2025}%
    (&Delta; = {nf_delta_total:+.2f} pp). The conflict did not create a new disability spike &mdash;
    it <strong>amplified existing vulnerabilities</strong>.
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 7: Distance Experiment (Q4) -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="distance">
  <h2 class="section-title">Physical Distance Is Not the Barrier</h2>
  <p class="section-subtitle">Q4: Distance to NII branches vs disability rate &mdash; SES slope = {trend_data['slope_ses']:+.2f}, Distance slope = {trend_data['slope_dist']:+.2f}</p>

  <div class="chart-card">
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;">
      <div class="chart-title" id="dist_chart_title" style="margin-bottom:0;">Disability Rate vs Distance to Nearest NII Branch (km)</div>
      <select id="dist_type_select" onchange="updateDistanceChart()" style="padding:6px 12px;border-radius:6px;border:1px solid #dee2e6;font-size:0.85rem;background:#fff;cursor:pointer;">
        <option value="dist_any">Any BTL Branch</option>
        <option value="dist_central">Central Branch</option>
      </select>
    </div>
    <div class="chart-wrap tall"><canvas id="chart_distance_scatter"></canvas></div>
  </div>

  <div class="chart-card">
    <div class="chart-title">Comparing Predictors on Same Scale: Socio-Economic Status (slope {trend_data['slope_ses']:+.2f}) vs Distance (slope {trend_data['slope_dist']:+.2f})</div>
    <div class="chart-wrap"><canvas id="chart_trend_lines"></canvas></div>
  </div>

  <div class="card-grid">
    <div class="card">
      <h3>What the Charts Show</h3>
      <p><strong>Top chart:</strong> each dot is a settlement. No visible trend &mdash;
      settlements far from NII branches have the same disability rates as those nearby.</p>
      <p style="margin-top:8px;"><strong>Bottom chart:</strong> two predictors on the same
      normalized scale. Socio-economic status has a strong slope ({trend_data['slope_ses']:+.2f}),
      while distance is nearly flat ({trend_data['slope_dist']:+.2f}).</p>
    </div>
    <div class="card">
      <h3>The Remaining Candidates</h3>
      <p>If not physical distance, then what?</p>
      <ul>
        <li><strong>Informational:</strong> people don't know they're eligible</li>
        <li><strong>Linguistic:</strong> forms and committees operate in Hebrew</li>
        <li><strong>Cultural:</strong> stigma prevents claiming</li>
        <li><strong>Institutional:</strong> the process itself deters applicants</li>
      </ul>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 8: Evidence -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="evidence">
  <h2 class="section-title">Israeli Research Confirms What Our Models Show</h2>
  <p class="section-subtitle">Two independent 2024 studies validate the access-barrier hypothesis</p>

  <div class="card-grid">
    <div class="card" style="border-top:3px solid var(--color-4);">
      <h3>Alhuzeel et al. (2024) &mdash; <em>Scandinavian J. of Disability Research</em></h3>
      <p style="font-style:italic;color:var(--color-4);font-weight:600;margin-bottom:8px;">
        &ldquo;It&rsquo;s Disgraceful Going through All this for Being an Arab and Disabled&rdquo;</p>
      <p>Interviews with 15 Arab Israelis with disabilities revealed barriers at every level:</p>
      <ul>
        <li><strong>Family:</strong> disability perceived as shameful; families avoid state services
            to prevent reduced marriage prospects</li>
        <li><strong>Institution:</strong> NII perceived as the <strong>most discriminatory</strong>
            government body. Medical committees lack Arabic-speaking staff</li>
        <li><strong>Society:</strong> &ldquo;the invisible of the invisible&rdquo; &mdash;
            Arab disabled people face compounded marginalization</li>
      </ul>
    </div>
    <div class="card" style="border-top:3px solid var(--color-1);">
      <h3>Brookdale Institute (2024) &mdash; People with Disabilities in the Arab Population</h3>
      <p>Arab self-reported disability prevalence is <strong>higher</strong> than Jewish
         &mdash; <strong style="color:var(--negative);">21% vs.&nbsp;19%</strong> &mdash; yet
         benefit claiming is <strong>lower</strong>.</p>
      <p style="margin-top:8px;">Arab people with disabilities face
         <strong>&ldquo;double exclusion&rdquo;</strong>. Women face
         <strong>&ldquo;triple exclusion&rdquo;</strong> (minority + disability + gender).</p>
      <p style="margin-top:8px;"><strong>35%</strong> of people already approved for
         benefits never exercised eligibility (Brookdale, 2022).
         <strong>23%</strong> were simply unaware of the approval.</p>
    </div>
  </div>

  <div class="card-grid" style="margin-top:var(--gap);">
    <div class="card" style="border-top:3px solid var(--color-5);">
      <h3>The Science Parallel: Obermeyer et al. (2019)</h3>
      <p><strong>Science, 366(6464)</strong>: A US healthcare algorithm trained on
         200&nbsp;million patients used <em>cost</em> as proxy for <em>need</em>.
         Because the system spent less on Black patients with equivalent illness,
         the algorithm concluded they were healthier.</p>
      <p style="margin-top:8px;"><strong>Our case is analogous:</strong> NII data records
         <em>benefit receipt</em> as proxy for <em>disability prevalence</em>. In communities
         facing access barriers, fewer people claim &mdash; creating the illusion of lower need.
         Our R&sup2; wall for Arab settlements ({arab_best_r2:.2f}) is statistical evidence of
         this same pattern.</p>
    </div>
    <div class="card" style="border-top:3px solid var(--color-3);">
      <h3>Convergence of Evidence</h3>
      <p>Our data-driven finding &mdash; that standard models fail for Arab settlements
         &mdash; aligns precisely with the qualitative research:</p>
      <ul>
        <li>The <strong>R&sup2; wall</strong> (models explain ~{arab_unexplained_pct}% less for Arab settlements) =
            unmeasured barriers</li>
        <li>The <strong>under-utilization clusters</strong> ({under_arab_pct}% are Arab-majority) =
            families not claiming despite need</li>
        <li>The <strong>distance non-effect</strong> (slope {trend_data['slope_dist']:+.2f}) =
            the barrier is not physical but informational, linguistic, and institutional</li>
      </ul>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 9: Recommendations -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="recommendations">
  <h2 class="section-title">Recommendations</h2>
  <p class="section-subtitle">Evidence-based actions to close the disability benefit access gap</p>

  <div class="rec-grid">
    <div class="rec-card">
      <div class="rec-num">1</div>
      <h4>Targeted Outreach</h4>
      <p>The {n_under} flagged settlements should receive proactive information
      campaigns &mdash; in Arabic, with community-based intermediaries and local
      advocacy organizations.</p>
    </div>
    <div class="rec-card">
      <div class="rec-num">2</div>
      <h4>Simplify the Process</h4>
      <p>Bhargava &amp; Manoli (2015): simplifying benefit language increased claiming
      by 6-8 pp. Show estimated benefit amounts, use plain language, reduce paperwork.</p>
    </div>
    <div class="rec-card">
      <div class="rec-num">3</div>
      <h4>Arabic-Language Services</h4>
      <p>Ensure NII medical committees include Arabic-speaking professionals.
      Provide all forms, notifications, and digital services in Arabic.</p>
    </div>
    <div class="rec-card">
      <div class="rec-num">4</div>
      <h4>Break the Intergenerational Cycle</h4>
      <p>Settlements flagged for both child and adult disability (Q2) need
      integrated family-level interventions, not separate programs.</p>
    </div>
    <div class="rec-card">
      <div class="rec-num">5</div>
      <h4>Annual Monitoring</h4>
      <p>Re-run the model annually. Settlements with persistent positive gaps
      across multiple years should be prioritized for field investigation.
      Track frontline communities for post-conflict effects.</p>
    </div>
  </div>

  <div class="insight-callout" style="margin-top:24px;">
    <strong>Summary:</strong> Four questions, one conclusion: disability benefit
    under-utilization in Israel is concentrated, intergenerational, structural,
    and invisible to standard data. Our models explain ~{round(best_r2_val * 100)}%
    of the variation overall (R&sup2; = {best_r2_val}) but hit a wall for Arab
    settlements (R&sup2; = {arab_best_r2:.2f}). This is evidence that the factors
    driving the gap &mdash; language barriers, institutional friction, and systemic
    under-investment &mdash; are invisible to official statistics.
  </div>
</div>

<footer style="text-align:center;padding:32px 16px 16px;font-size:0.82rem;color:var(--text-secondary);">
  Data as of: December 2024 (NII) &nbsp;|&nbsp; TovTech Research Group &nbsp;|&nbsp; Generated by generate_presentation_experimental.py
</footer>

</div><!-- /dashboard -->

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Chart.js Scripts -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<script>
// ── Data declarations ──
const olsCoeffData = {json.dumps(ols_coeff_data)};
const importanceData = {json.dumps(importance_data)};
const olsAnomalyData = {json.dumps(ols_anomaly_data)};
const olsMinVal = {round(ols_min_val, 2)};
const olsMaxVal = {round(ols_max_val, 2)};
const olsThreshold = {round(ols_threshold, 4)};
const interScatterData = {json.dumps(inter_scatter_data)};
const arabChartData = {json.dumps(arab_chart_data)};
const temporalChartData = {json.dumps(temporal_chart_data)};
const distScatterData = {json.dumps(dist_scatter_data)};
const trendData = {json.dumps(trend_data)};
const gapScatterData = {json.dumps(gap_scatter_data)};

// ── Tooltip defaults ──
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.tooltip.backgroundColor = "rgba(33,37,41,0.92)";
Chart.defaults.plugins.tooltip.titleFont = {{ weight: "600" }};
Chart.defaults.plugins.tooltip.bodyFont = {{ size: 12 }};
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 6;

// ── 1. OLS Coefficient Horizontal Bar ──
new Chart(document.getElementById("chart_ols_coeff"), {{
  type: "bar",
  data: {{
    labels: olsCoeffData.labels,
    datasets: [{{
      label: "Standardized Coefficient",
      data: olsCoeffData.values,
      backgroundColor: olsCoeffData.colors,
      borderColor: olsCoeffData.colors,
      borderWidth: 1,
      borderRadius: 4,
      barPercentage: 0.7
    }}]
  }},
  options: {{
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => "Coefficient: " + ctx.parsed.x.toFixed(4)
        }}
      }}
    }},
    scales: {{
      x: {{
        grid: {{ color: "rgba(0,0,0,0.06)" }},
        title: {{ display: true, text: "Standardized Coefficient", font: {{ weight: "600" }} }}
      }},
      y: {{
        grid: {{ display: false }},
        ticks: {{ font: {{ size: 11 }} }}
      }}
    }}
  }}
}});

// ── 2. RF/XGB Feature Importance Horizontal Bar ──
new Chart(document.getElementById("chart_importance"), {{
  type: "bar",
  data: {{
    labels: importanceData.labels,
    datasets: [{{
      label: "Importance",
      data: importanceData.values,
      backgroundColor: importanceData.values.map((v, i) => {{
        const maxV = Math.max(...importanceData.values);
        const ratio = v / maxV;
        return ratio > 0.7 ? "#DD8452" : ratio > 0.4 ? "#4C72B0" : "#adb5bd";
      }}),
      borderRadius: 4,
      barPercentage: 0.7
    }}]
  }},
  options: {{
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => "Importance: " + ctx.parsed.x.toFixed(4)
        }}
      }}
    }},
    scales: {{
      x: {{
        grid: {{ color: "rgba(0,0,0,0.06)" }},
        title: {{ display: true, text: "Feature Importance", font: {{ weight: "600" }} }}
      }},
      y: {{
        grid: {{ display: false }},
        ticks: {{ font: {{ size: 11 }} }}
      }}
    }}
  }}
}});

// ── 2b. Ensemble Gap Scatter (Actual vs Expected, 4 segments) ──
(function() {{
  const segCfg = {{
    "Both Lower":                {{ color: "rgba(148,163,184,0.4)", border: "#94A3B8", radius: 5, style: "circle" }},
    "Both High":                 {{ color: "rgba(129,140,248,0.5)", border: "#818CF8", radius: 6, style: "circle" }},
    "Potential Under-utilization":{{ color: "rgba(245,158,11,0.85)", border: "#F59E0B", radius: 9, style: "rectRot" }},
    "Hidden Burden":             {{ color: "rgba(244,63,94,0.85)",  border: "#F43F5E",  radius: 9, style: "crossRot" }}
  }};
  const order = ["Both Lower", "Both High", "Potential Under-utilization", "Hidden Burden"];
  const datasets = order.filter(seg => gapScatterData[seg] && gapScatterData[seg].x.length > 0).map(seg => {{
    const d = gapScatterData[seg];
    const cfg = segCfg[seg];
    return {{
      label: seg,
      data: d.x.map((xv, i) => ({{ x: xv, y: d.y[i], name: d.names[i], gap: d.gap[i] }})),
      backgroundColor: cfg.color,
      borderColor: cfg.border,
      pointRadius: cfg.radius,
      pointHoverRadius: cfg.radius + 3,
      pointStyle: cfg.style
    }};
  }});

  // Compute diagonal range
  const allX = order.flatMap(s => gapScatterData[s] ? gapScatterData[s].x : []);
  const allY = order.flatMap(s => gapScatterData[s] ? gapScatterData[s].y : []);
  const diagMin = Math.min(...allX, ...allY);
  const diagMax = Math.max(...allX, ...allY);

  const gapDiagonalPlugin = {{
    id: "gapDiagonal",
    afterDraw(chart) {{
      const {{ ctx, scales: {{ x: xScale, y: yScale }} }} = chart;
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = "rgba(244, 63, 94, 0.5)";
      ctx.lineWidth = 1.5;
      ctx.moveTo(xScale.getPixelForValue(diagMin), yScale.getPixelForValue(diagMin));
      ctx.lineTo(xScale.getPixelForValue(diagMax), yScale.getPixelForValue(diagMax));
      ctx.stroke();
      ctx.restore();
    }}
  }};

  new Chart(document.getElementById("chart_gap_scatter"), {{
    type: "scatter",
    data: {{ datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{
          position: "top",
          labels: {{ usePointStyle: true, padding: 16 }}
        }},
        tooltip: {{
          callbacks: {{
            label: ctx => {{
              const pt = ctx.raw;
              return [
                pt.name || "",
                "Actual: " + pt.x.toFixed(2) + "%",
                "Expected: " + pt.y.toFixed(2) + "%",
                "Gap: " + (pt.gap >= 0 ? "+" : "") + pt.gap.toFixed(2) + " pp"
              ];
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: "Actual Rate (%)", font: {{ weight: "600" }} }},
          grid: {{ color: "rgba(0,0,0,0.06)" }}
        }},
        y: {{
          title: {{ display: true, text: "Expected Rate (%)", font: {{ weight: "600" }} }},
          grid: {{ color: "rgba(0,0,0,0.06)" }}
        }}
      }}
    }},
    plugins: [gapDiagonalPlugin]
  }});
}})();

// ── 3. OLS Anomaly Scatter (Actual vs Predicted) ──
(function() {{
  const datasets = [];
  const colorMap = {{ "Normal": "#4C72B0", "Underutilization": "#55A868", "Dependency": "#C44E52" }};
  const shapeMap = {{ "Normal": "circle", "Underutilization": "rectRot", "Dependency": "crossRot" }};
  const sizeMap = {{ "Normal": 5, "Underutilization": 8, "Dependency": 8 }};

  ["Normal", "Underutilization", "Dependency"].forEach(atype => {{
    const d = olsAnomalyData[atype];
    if (!d || d.predicted.length === 0) return;
    datasets.push({{
      label: atype,
      data: d.predicted.map((p, i) => ({{ x: p, y: d.actual[i], name: d.names[i], res: d.residual[i] }})),
      backgroundColor: colorMap[atype],
      borderColor: colorMap[atype],
      pointStyle: shapeMap[atype],
      pointRadius: sizeMap[atype],
      pointHoverRadius: sizeMap[atype] + 3
    }});
  }});

  // Diagonal line plugin
  const diagonalPlugin = {{
    id: "diagonalLine",
    afterDraw: function(chart) {{
      const ctx = chart.ctx;
      const xScale = chart.scales.x;
      const yScale = chart.scales.y;
      const minV = Math.max(xScale.min, yScale.min);
      const maxV = Math.min(xScale.max, yScale.max);

      ctx.save();
      // Main diagonal
      ctx.beginPath();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = "rgba(0,0,0,0.3)";
      ctx.lineWidth = 1.5;
      ctx.moveTo(xScale.getPixelForValue(minV), yScale.getPixelForValue(minV));
      ctx.lineTo(xScale.getPixelForValue(maxV), yScale.getPixelForValue(maxV));
      ctx.stroke();

      // Threshold lines
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      // Upper
      ctx.strokeStyle = "rgba(196,78,82,0.3)";
      ctx.beginPath();
      ctx.moveTo(xScale.getPixelForValue(minV), yScale.getPixelForValue(minV + olsThreshold));
      ctx.lineTo(xScale.getPixelForValue(maxV), yScale.getPixelForValue(maxV + olsThreshold));
      ctx.stroke();
      // Lower
      ctx.strokeStyle = "rgba(85,168,104,0.3)";
      ctx.beginPath();
      ctx.moveTo(xScale.getPixelForValue(minV), yScale.getPixelForValue(minV - olsThreshold));
      ctx.lineTo(xScale.getPixelForValue(maxV), yScale.getPixelForValue(maxV - olsThreshold));
      ctx.stroke();

      ctx.restore();
    }}
  }};

  new Chart(document.getElementById("chart_anomaly_scatter"), {{
    type: "scatter",
    data: {{ datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{
          position: "top",
          labels: {{ usePointStyle: true, padding: 16 }}
        }},
        tooltip: {{
          callbacks: {{
            label: ctx => {{
              const pt = ctx.raw;
              return [
                pt.name || "",
                "Actual: " + pt.y.toFixed(2) + "%",
                "Predicted: " + pt.x.toFixed(2) + "%",
                "Residual: " + (pt.res >= 0 ? "+" : "") + pt.res.toFixed(2)
              ];
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: "Predicted Rate (%)", font: {{ weight: "600" }} }},
          grid: {{ color: "rgba(0,0,0,0.06)" }}
        }},
        y: {{
          title: {{ display: true, text: "Actual Rate (%)", font: {{ weight: "600" }} }},
          grid: {{ color: "rgba(0,0,0,0.06)" }}
        }}
      }}
    }},
    plugins: [diagonalPlugin]
  }});
}})();

// ── 4. Intergenerational Scatter ──
(function() {{
  const datasets = interScatterData.groups.map(g => ({{
    label: g.name,
    data: g.child_rate.map((c, i) => ({{ x: c, y: g.adult_rate[i], name: g.names[i] }})),
    backgroundColor: g.color,
    borderColor: g.color,
    pointRadius: 5,
    pointHoverRadius: 8
  }}));

  new Chart(document.getElementById("chart_intergenerational"), {{
    type: "scatter",
    data: {{ datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{
          position: "top",
          labels: {{ usePointStyle: true, padding: 16 }}
        }},
        tooltip: {{
          callbacks: {{
            label: ctx => {{
              const pt = ctx.raw;
              return [
                pt.name || "",
                "Child rate: " + pt.x.toFixed(2) + "%",
                "Adult rate: " + pt.y.toFixed(2) + "%"
              ];
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: "Child Disability Rate (%)", font: {{ weight: "600" }} }},
          grid: {{ color: "rgba(0,0,0,0.06)" }}
        }},
        y: {{
          title: {{ display: true, text: "Adult Disability Rate (%)", font: {{ weight: "600" }} }},
          grid: {{ color: "rgba(0,0,0,0.06)" }}
        }}
      }}
    }}
  }});
}})();

// ── 5. Arab R2 Grouped Bar ──
(function() {{
  new Chart(document.getElementById("chart_arab_r2"), {{
    type: "bar",
    data: {{
      labels: arabChartData.models,
      datasets: [
        {{
          label: "Secular",
          data: arabChartData.secular_r2,
          backgroundColor: "#4C72B0",
          borderRadius: 4,
          barPercentage: 0.8,
          categoryPercentage: 0.7
        }},
        {{
          label: "All 278",
          data: arabChartData.overall_r2,
          backgroundColor: "#adb5bd",
          borderRadius: 4,
          barPercentage: 0.8,
          categoryPercentage: 0.7
        }},
        {{
          label: "Arab >50%",
          data: arabChartData.arab_r2,
          backgroundColor: "#C44E52",
          borderRadius: 4,
          barPercentage: 0.8,
          categoryPercentage: 0.7
        }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{
          position: "top",
          labels: {{ padding: 16 }}
        }},
        tooltip: {{
          callbacks: {{
            label: ctx => ctx.dataset.label + ": R\\u00b2 = " + (ctx.parsed.y !== null ? ctx.parsed.y.toFixed(3) : "N/A")
          }}
        }}
      }},
      scales: {{
        y: {{
          beginAtZero: true,
          max: 1,
          title: {{ display: true, text: "R\\u00b2 (higher = better)", font: {{ weight: "600" }} }},
          grid: {{ color: "rgba(0,0,0,0.06)" }}
        }},
        x: {{
          grid: {{ display: false }}
        }}
      }}
    }}
  }});
}})();

// ── 6. Temporal Line Chart ──
(function() {{
  const colors = ["#C44E52", "#4C72B0"];
  const datasets = temporalChartData.series.map((s, i) => ({{
    label: s.name,
    data: s.rates,
    borderColor: colors[i],
    backgroundColor: colors[i],
    pointRadius: 6,
    pointHoverRadius: 9,
    borderWidth: 3,
    tension: 0,
    fill: false
  }}));

  new Chart(document.getElementById("chart_temporal"), {{
    type: "line",
    data: {{
      labels: temporalChartData.series[0].years.map(String),
      datasets
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{
          position: "top",
          labels: {{ padding: 16 }}
        }},
        tooltip: {{
          callbacks: {{
            label: ctx => ctx.dataset.label + ": " + ctx.parsed.y.toFixed(2) + "%"
          }}
        }}
      }},
      scales: {{
        y: {{
          title: {{ display: true, text: "Disability Rate (%)", font: {{ weight: "600" }} }},
          grid: {{ color: "rgba(0,0,0,0.06)" }}
        }},
        x: {{
          title: {{ display: true, text: "Year", font: {{ weight: "600" }} }},
          grid: {{ display: false }}
        }}
      }}
    }}
  }});
}})();

// ── 7. Distance Scatter (with dropdown) ──
let distChart;
const distLabels = {{ dist_any: "Any BTL Branch", dist_central: "Central Branch" }};
const distColors = {{ dist_any: ["rgba(76, 114, 176, 0.5)", "#4C72B0"], dist_central: ["rgba(244, 63, 94, 0.5)", "#F43F5E"] }};

function buildDistData(key) {{
  return distScatterData[key].map((d, i) => ({{
    x: d,
    y: distScatterData.disability_rate[i],
    name: distScatterData.names[i]
  }}));
}}

function updateDistanceChart() {{
  const key = document.getElementById("dist_type_select").value;
  distChart.data.datasets[0].data = buildDistData(key);
  distChart.data.datasets[0].backgroundColor = distColors[key][0];
  distChart.data.datasets[0].borderColor = distColors[key][1];
  distChart.options.scales.x.title.text = "Distance to Nearest " + distLabels[key] + " (km)";
  document.getElementById("dist_chart_title").textContent = "Disability Rate vs Distance to " + distLabels[key] + " (km)";
  distChart.update("none");
}}

(function() {{
  distChart = new Chart(document.getElementById("chart_distance_scatter"), {{
    type: "scatter",
    data: {{
      datasets: [{{
        label: "Settlement",
        data: buildDistData("dist_any"),
        backgroundColor: distColors.dist_any[0],
        borderColor: distColors.dist_any[1],
        pointRadius: 5,
        pointHoverRadius: 8
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: ctx => {{
              const pt = ctx.raw;
              return [
                pt.name || "",
                "Disability rate: " + pt.y.toFixed(2) + "%",
                "Distance: " + pt.x.toFixed(1) + " km"
              ];
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: "Distance to Nearest Any BTL Branch (km)", font: {{ weight: "600" }} }},
          grid: {{ color: "rgba(0,0,0,0.06)" }},
          beginAtZero: true
        }},
        y: {{
          title: {{ display: true, text: "Disability Rate (%)", font: {{ weight: "600" }} }},
          grid: {{ color: "rgba(0,0,0,0.06)" }}
        }}
      }}
    }}
  }});
}})();

// ── 8. Trend Lines (SES vs Distance) ──
(function() {{
  new Chart(document.getElementById("chart_trend_lines"), {{
    type: "line",
    data: {{
      labels: trendData.x_grid,
      datasets: [
        {{
          label: "Socio-economic status (slope {trend_data['slope_ses']:+.2f})",
          data: trendData.y_ses,
          borderColor: "#C44E52",
          backgroundColor: "rgba(196,78,82,0.1)",
          borderWidth: 3,
          pointRadius: 0,
          fill: true,
          tension: 0.1
        }},
        {{
          label: "Distance to NII branch (slope {trend_data['slope_dist']:+.2f})",
          data: trendData.y_dist,
          borderColor: "#4C72B0",
          backgroundColor: "rgba(76,114,176,0.05)",
          borderWidth: 3,
          pointRadius: 0,
          fill: true,
          tension: 0.1
        }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{
          position: "top",
          labels: {{ padding: 16 }}
        }},
        tooltip: {{
          mode: "index",
          intersect: false,
          callbacks: {{
            title: items => "z = " + parseFloat(items[0].label).toFixed(2),
            label: ctx => ctx.dataset.label.split(" (")[0] + ": " + ctx.parsed.y.toFixed(2) + "%"
          }}
        }}
      }},
      scales: {{
        x: {{
          type: "linear",
          title: {{ display: true, text: "Standardized Scale (low to high)", font: {{ weight: "600" }} }},
          grid: {{ color: "rgba(0,0,0,0.06)" }},
          ticks: {{
            callback: function(value) {{ return value.toFixed(1); }},
            maxTicksLimit: 10
          }}
        }},
        y: {{
          title: {{ display: true, text: "Disability Rate (%)", font: {{ weight: "600" }} }},
          grid: {{ color: "rgba(0,0,0,0.06)" }}
        }}
      }}
    }}
  }});
}})();

// ── Table sorting ──
function sortTable(tableId, colIdx) {{
  const table = document.getElementById(tableId);
  const tbody = table.querySelector("tbody");
  const rows = Array.from(tbody.querySelectorAll("tr"));
  const th = table.querySelectorAll("th")[colIdx];

  // Toggle sort direction
  const isAsc = th.dataset.sort !== "asc";
  table.querySelectorAll("th").forEach(h => {{ h.dataset.sort = ""; h.classList.remove("sorted"); }});
  th.dataset.sort = isAsc ? "asc" : "desc";
  th.classList.add("sorted");

  rows.sort((a, b) => {{
    let aVal = a.cells[colIdx].textContent.trim();
    let bVal = b.cells[colIdx].textContent.trim();
    // Try numeric
    const aNum = parseFloat(aVal.replace(/[%,+]/g, ""));
    const bNum = parseFloat(bVal.replace(/[%,+]/g, ""));
    if (!isNaN(aNum) && !isNaN(bNum)) {{
      return isAsc ? aNum - bNum : bNum - aNum;
    }}
    return isAsc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
  }});

  rows.forEach(r => tbody.appendChild(r));
}}

// ── Scroll-spy for navigation ──
(function() {{
  const links = document.querySelectorAll(".top-nav .nav-links a");
  const sections = document.querySelectorAll(".section");

  const observer = new IntersectionObserver(entries => {{
    entries.forEach(entry => {{
      if (entry.isIntersecting) {{
        const id = entry.target.id;
        links.forEach(link => {{
          link.classList.toggle("active", link.getAttribute("href") === "#" + id);
        }});
      }}
    }});
  }}, {{ root: null, rootMargin: "-40% 0px -50% 0px", threshold: 0 }});

  sections.forEach(s => observer.observe(s));
}})();
</script>

</body>
</html>
"""

# ── Write output ──────────────────────────────────────────────────────────────

out_path = PROJECT_ROOT / "presentation_experimental.html"
out_path.write_text(html_output, encoding="utf-8")
print(f"\nDone! Written to {out_path}")
print(f"  File size: {out_path.stat().st_size / 1024:.0f} KB")
