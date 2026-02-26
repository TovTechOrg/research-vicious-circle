"""
Generate EXPERIMENTAL HTML dashboard presentation.
Light-themed, Chart.js-based alternative to the dark Plotly executive presentation.
Reads benefits_final.csv, trains the same models, loads temporal + distance data,
produces presentation_experimental.html.

Sections (7 narrative-driven):
  1. The Problem — human-first overview with TL;DR
  2. Missing Claimants — Q1 ensemble gap scatter
  3. Childhood Cycle — Q2 intergenerational
  4. The Wall — climax, Arab sector R2 + integrated evidence
  5. Ruled Out — Q3 + Q4 combined (conflict + distance)
  6. Methodology — OLS + RF/XGB for technical readers
  7. Action Plan — 5 recommendations tied to findings
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
from scipy.stats import spearmanr, pearsonr, norm as sp_norm
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
ols_valid_idx = df_ols_clean.index  # save for sector R2 later
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

# OLS group breakdown (Arab / Haredi / Other)
ols_arab_pct_vals = pd.to_numeric(
    df_ols_clean.get("arab_population_percentage", pd.Series(dtype=float)), errors="coerce"
).fillna(0).values
ols_haredi_pct_vals = pd.to_numeric(
    df_ols_clean.get("haredi_population_percentage", pd.Series(dtype=float)), errors="coerce"
).fillna(0).values
ols_under_idx = [i for i, t in enumerate(anomaly_types) if t == "Underutilization"]
n_ols_arab_under = sum(1 for i in ols_under_idx if ols_arab_pct_vals[i] > 50)
n_ols_haredi_under = sum(1 for i in ols_under_idx if ols_haredi_pct_vals[i] > 50)
n_ols_other_under = n_ols_under - n_ols_arab_under - n_ols_haredi_under
ols_under_arab_pct = round(n_ols_arab_under / n_ols_under * 100) if n_ols_under > 0 else 0
ols_under_haredi_pct = round(n_ols_haredi_under / n_ols_under * 100) if n_ols_under > 0 else 0
ols_under_other_pct = round(n_ols_other_under / n_ols_under * 100) if n_ols_under > 0 else 0

# OLS status lookup: index → anomaly type (for explorer table)
ols_status_map = dict(zip(df_ols_clean.index, anomaly_types))

# ── Sector R2 ───────────────────────────────────────────────────────────────

print("Computing sector R2 ...")

arab_pct = df_reg["arab_population_percentage"].values
haredi_pct = df_reg["haredi_population_percentage"].values
arab_mask = arab_pct > 50
haredi_mask = haredi_pct > 50
secular_mask = (~arab_mask) & (~haredi_mask)

sector_rows = []
# OLS full-fit predictions mapped to df_reg index
ols_full_pred_arr = np.full(len(y), np.nan)
ols_valid_positions = np.array([list(df_reg.index).index(idx) for idx in ols_valid_idx])
ols_full_pred_arr[ols_valid_positions] = ols_model.predict(scaler.transform(X_ols_raw))

for sname, smask in [
    ("Arab >50%", arab_mask), ("Haredi >50%", haredi_mask),
    ("Secular", secular_mask), ("All 278", np.ones(len(y), dtype=bool)),
]:
    n = int(smask.sum())
    row = {"sector": sname, "n": n}
    row["rf_r2"] = round(r2_score(y[smask], rf_pred_oof[smask]), 3) if n >= 5 else None
    row["xgb_r2"] = round(r2_score(y[smask], xgb_pred_oof[smask]), 3) if xgb_available and n >= 5 else None
    row["tabpfn_r2"] = round(r2_score(y[smask], tabpfn_pred_oof[smask]), 3) if tabpfn_available and n >= 5 else None
    # OLS sector R2 (only where OLS has predictions)
    ols_sector_mask = smask & ~np.isnan(ols_full_pred_arr)
    n_ols_sector = int(ols_sector_mask.sum())
    row["ols_r2"] = round(r2_score(y[ols_sector_mask], ols_full_pred_arr[ols_sector_mask]), 3) if n_ols_sector >= 5 else None
    sector_rows.append(row)

arab_r2_vals = [v for row in sector_rows if row["sector"] == "Arab >50%"
                for k, v in row.items() if k.endswith("_r2") and k != "ols_r2" and v is not None]
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

# Add Linear Regression to sector comparison
if arab_row.get("ols_r2") is not None:
    arab_chart_data["models"].append("Linear Regression")
    arab_chart_data["arab_r2"].append(arab_row["ols_r2"])
    arab_chart_data["overall_r2"].append(all_row.get("ols_r2"))
    arab_chart_data["secular_r2"].append(secular_row.get("ols_r2"))

best_model = "TabPFN v2" if tabpfn_available else ("XGBoost" if xgb_available else "RandomForest")
best_r2_val = (tabpfn_scores or xgb_scores or rf_scores)["r2"]

# ── Counterfactual analysis: train on non-Arab, predict Arab ─────────────────

print("Computing counterfactual analysis ...")

cf_arab_mask = arab_mask  # already defined: arab_pct > 50
cf_non_arab_mask = ~cf_arab_mask

X_non_arab = X[cf_non_arab_mask]
y_non_arab = y[cf_non_arab_mask]
X_arab_cf = X[cf_arab_mask]
y_arab_cf = y[cf_arab_mask]

# Train ensemble on non-Arab only
cf_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestRegressor(n_estimators=600, min_samples_leaf=2, random_state=42, n_jobs=1)),
])
cf_rf.fit(X_non_arab, y_non_arab)
cf_pred = cf_rf.predict(X_arab_cf)

if xgb_available:
    cf_xgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", XGBRegressor(n_estimators=450, max_depth=4, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9,
                             objective="reg:squarederror", random_state=42, n_jobs=1)),
    ])
    cf_xgb.fit(X_non_arab, y_non_arab)
    cf_pred = (cf_pred + cf_xgb.predict(X_arab_cf)) / 2.0

if tabpfn_available:
    cf_tabpfn = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("tabpfn", TabPFNRegressor.create_default_for_version("v2")),
    ])
    try:
        cf_tabpfn.fit(X_non_arab, y_non_arab)
        n_cf_models = 2 if not xgb_available else 3
        cf_pred = (cf_pred * (n_cf_models - 1) + cf_tabpfn.predict(X_arab_cf)) / n_cf_models
    except Exception:
        pass

cf_gap = cf_pred - y_arab_cf  # positive = under-utilization
cf_names = df_reg.loc[df_reg.index[cf_arab_mask], "settlement_name"].tolist()
cf_pop_18_64 = pd.to_numeric(
    df.loc[df_reg.index[cf_arab_mask], "population_18_64"], errors="coerce"
).values

# Threshold: top quartile of gap (biggest under-utilization)
cf_gap_threshold = float(np.percentile(cf_gap[cf_gap > 0], 75)) if (cf_gap > 0).sum() > 4 else 2.0

cf_flagged_mask = cf_gap >= cf_gap_threshold
n_cf_flagged = int(cf_flagged_mask.sum())
n_cf_under = int((cf_gap > 0).sum())
cf_mean_gap = round(float(np.mean(cf_gap)), 2)
cf_median_gap = round(float(np.median(cf_gap)), 2)

# Estimate missing beneficiaries
cf_missing_people = int(np.sum(
    np.where(cf_gap > 0, (cf_gap / 100) * cf_pop_18_64, 0)
))

# Build scatter data for chart (all Arab settlements: actual vs counterfactual)
cf_scatter = {
    "flagged": {
        "x": [round(float(cf_pred[i]), 2) for i in range(len(cf_gap)) if cf_flagged_mask[i]],
        "y": [round(float(y_arab_cf[i]), 2) for i in range(len(cf_gap)) if cf_flagged_mask[i]],
        "names": [cf_names[i] for i in range(len(cf_gap)) if cf_flagged_mask[i]],
        "gap": [round(float(cf_gap[i]), 2) for i in range(len(cf_gap)) if cf_flagged_mask[i]],
    },
    "normal": {
        "x": [round(float(cf_pred[i]), 2) for i in range(len(cf_gap)) if not cf_flagged_mask[i]],
        "y": [round(float(y_arab_cf[i]), 2) for i in range(len(cf_gap)) if not cf_flagged_mask[i]],
        "names": [cf_names[i] for i in range(len(cf_gap)) if not cf_flagged_mask[i]],
        "gap": [round(float(cf_gap[i]), 2) for i in range(len(cf_gap)) if not cf_flagged_mask[i]],
    },
}
cf_min_val = round(float(min(min(cf_pred), min(y_arab_cf))) - 0.5, 1)
cf_max_val = round(float(max(max(cf_pred), max(y_arab_cf))) + 0.5, 1)

print(f"  {n_cf_under} of {len(cf_gap)} Arab settlements receive less than counterfactual")
print(f"  Mean gap: {cf_mean_gap} pp, Median gap: {cf_median_gap} pp")
print(f"  Flagged (top quartile, gap >= {cf_gap_threshold:.1f} pp): {n_cf_flagged}")
print(f"  Estimated missing beneficiaries: ~{cf_missing_people:,}")

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

# Residual-based segmentation (±1.5σ from diagonal — consistent with OLS approach)
gap_std = float(np.std(df_gap["gap_pp"]))
gap_threshold = 1.5 * gap_std  # ≈ 1.95 pp


def _segment(row: pd.Series) -> str:
    gap = row["gap_pp"]  # positive = model expects more than actual (under-utilization)
    if gap >= gap_threshold:
        return "Potential Under-utilization"
    if gap <= -gap_threshold:
        return "Hidden Burden"
    return "Normal"


df_gap["segment"] = df_gap.apply(_segment, axis=1)

segment_order = ["Normal", "Potential Under-utilization", "Hidden Burden"]
gap_scatter_data: dict[str, dict] = {}
for seg in segment_order:
    sub = df_gap[df_gap["segment"] == seg]
    gap_scatter_data[seg] = {
        "x": sub["mean_pred_rate"].round(2).tolist(),
        "y": sub["actual_rate"].round(2).tolist(),
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
arab_hidden = hidden_candidates[arab_pct_col.reindex(hidden_candidates.index).fillna(0) > 50]
n_arab = int(arab_mask.sum())
n_arab_under = len(arab_under)
n_arab_hidden = len(arab_hidden)
under_arab_pct = round(len(arab_under) / n_under * 100) if n_under > 0 else 0
hidden_arab_pct = round(len(arab_hidden) / n_hidden * 100) if n_hidden > 0 else 0

# Haredi breakdown
haredi_pct_col = pd.to_numeric(
    df_gap.get("haredi_population_percentage", pd.Series(dtype=float)), errors="coerce"
).fillna(0)
haredi_under = under_candidates[haredi_pct_col.reindex(under_candidates.index).fillna(0) > 50]
haredi_hidden = hidden_candidates[haredi_pct_col.reindex(hidden_candidates.index).fillna(0) > 50]
n_haredi_under = len(haredi_under)
n_haredi_hidden = len(haredi_hidden)
under_haredi_pct = round(n_haredi_under / n_under * 100) if n_under > 0 else 0
hidden_haredi_pct = round(n_haredi_hidden / n_hidden * 100) if n_hidden > 0 else 0

# Other (secular/mixed)
n_other_under = n_under - n_arab_under - n_haredi_under
n_other_hidden = n_hidden - n_arab_hidden - n_haredi_hidden
under_other_pct = round(n_other_under / n_under * 100) if n_under > 0 else 0

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

# Pearson for overall
r_overall, p_r_overall = pearsonr(
    child_rate_col[inter_valid], adult_rate_col[inter_valid]
)
r_overall = round(r_overall, 3)
r2_inter = round(r_overall ** 2 * 100, 1)

# Periphery split: 1-5 = peripheral, 6-10 = non-peripheral (notebook convention)
periph_periphery = periph_cluster.isin([1, 2, 3, 4, 5])
periph_nonperiph = periph_cluster.isin([6, 7, 8, 9, 10])
# Keep old 3-tier for scatter colours
periph_outer = periph_cluster.isin([1, 2, 3])
periph_center = periph_cluster.isin([8, 9, 10])

mask_periph = inter_valid & periph_periphery
rho_periph, _ = spearmanr(child_rate_col[mask_periph], adult_rate_col[mask_periph])
r_periph, p_r_periph = pearsonr(child_rate_col[mask_periph], adult_rate_col[mask_periph])
rho_periph = round(rho_periph, 3)
r_periph = round(r_periph, 3)
n_periph = int(mask_periph.sum())

mask_nonperiph = inter_valid & periph_nonperiph
rho_nonperiph, _ = spearmanr(child_rate_col[mask_nonperiph], adult_rate_col[mask_nonperiph])
r_nonperiph, p_r_nonperiph = pearsonr(child_rate_col[mask_nonperiph], adult_rate_col[mask_nonperiph])
rho_nonperiph = round(rho_nonperiph, 3)
r_nonperiph = round(r_nonperiph, 3)
n_nonperiph = int(mask_nonperiph.sum())

# Keep center (8-10) stats for display
mask_center = inter_valid & periph_center
rho_center, _ = spearmanr(child_rate_col[mask_center], adult_rate_col[mask_center])
rho_center = round(rho_center, 3)
n_center = int(mask_center.sum())

n_inter = int(inter_valid.sum())

# Fisher's Z test (Pearson: peripheral vs non-peripheral)
def _fisher_z(r1: float, n1: int, r2: float, n2: int):
    if any(abs(r) >= 1 or k < 4 for r, k in [(r1, n1), (r2, n2)]):
        return float("nan"), float("nan")
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - sp_norm.cdf(abs(z)))
    return round(z, 3), round(p, 4)

fisher_z, fisher_p = _fisher_z(r_periph, n_periph, r_nonperiph, n_nonperiph)
fisher_sig = "p < 0.05" if fisher_p < 0.05 else f"p = {fisher_p}"

# High-Risk Cluster: both adult + child > P75
gdr_p75 = float(adult_rate_col[inter_valid].quantile(0.75))
dcr_p75 = float(child_rate_col[inter_valid].quantile(0.75))
hr_mask = (adult_rate_col > gdr_p75) & (child_rate_col > dcr_p75) & inter_valid
n_high_risk = int(hr_mask.sum())
high_risk_pct = round(n_high_risk / n_inter * 100, 1)

# Compare SES medians: high-risk vs rest
ses_col = pd.to_numeric(df.loc[df_reg.index, "socio_economic_index_score"], errors="coerce")
sal_col = pd.to_numeric(df.loc[df_reg.index, "average_monthly_salary_2023"], errors="coerce")
ses_hr = ses_col[hr_mask].median()
ses_rest = ses_col[inter_valid & ~hr_mask].median()
sal_hr = sal_col[hr_mask].median()
sal_rest = sal_col[inter_valid & ~hr_mask].median()

# High-risk cluster boxplot data + Welch's T-test
HR_INDICATORS = [
    ("socio_economic_index_score", "Socio-Economic Score", "scale ~-3 to +3"),
    ("peripherality_index_score", "Peripherality Score", "lower = more peripheral"),
    ("average_monthly_salary_2023", "Avg Salary (NIS)", "monthly"),
    ("income_support_rate", "Income Support Rate", "% of working-age pop."),
    ("edu_bagrut_eligibility_pct", "Bagrut Eligibility", "% of students"),
    ("edu_attain_pct_academic_degree", "Academic Degree", "% of pop. 25-65"),
]
hr_boxplot_data = {"indicators": [], "high_risk": [], "rest": []}
hr_comparison = []  # For detailed comparison cards
from scipy.stats import ttest_ind
for col_name, label, unit in HR_INDICATORS:
    if col_name not in df.columns:
        continue
    vals = pd.to_numeric(df.loc[df_reg.index, col_name], errors="coerce")
    hr_vals = vals[hr_mask].dropna()
    rest_vals = vals[inter_valid & ~hr_mask].dropna()
    hr_boxplot_data["indicators"].append(label)
    hr_boxplot_data["high_risk"].append([round(float(v), 2) for v in hr_vals.tolist()])
    hr_boxplot_data["rest"].append([round(float(v), 2) for v in rest_vals.tolist()])
    # Welch's T-test
    if len(hr_vals) >= 2 and len(rest_vals) >= 2:
        t_stat, p_val = ttest_ind(hr_vals.values, rest_vals.values, equal_var=False)
    else:
        t_stat, p_val = float("nan"), float("nan")
    hr_mean = float(hr_vals.mean())
    rest_mean = float(rest_vals.mean())
    delta = hr_mean - rest_mean
    hr_comparison.append({
        "col": col_name, "label": label, "unit": unit,
        "hr_mean": hr_mean, "rest_mean": rest_mean, "delta": delta,
        "t_stat": t_stat, "p_val": p_val,
        "significant": p_val < 0.05 if not np.isnan(p_val) else False,
    })

n_significant = sum(1 for c in hr_comparison if c["significant"])
print(f"  Welch's T-test: {n_significant}/{len(hr_comparison)} indicators significant (p<0.05)")

# Scatter data for Chart.js — dual panel (peripheral vs non-peripheral)
ses_cluster_col = pd.to_numeric(df.loc[df_reg.index, "socio_economic_index_cluster"], errors="coerce")
pop_col = pd.to_numeric(df.loc[df_reg.index, "total_population"], errors="coerce").fillna(5000)

from scipy.stats import linregress as _linregress

inter_scatter_data = {"groups": []}
for group_label, group_mask, color, grp_r, grp_p, grp_n in [
    ("Peripheral (1-5)", periph_periphery & inter_valid, "#DD8452", r_periph, p_r_periph, n_periph),
    ("Non-Peripheral (6-10)", periph_nonperiph & inter_valid, "#4C72B0", r_nonperiph, p_r_nonperiph, n_nonperiph),
]:
    sub_child = child_rate_col[group_mask].round(2).tolist()
    sub_adult = adult_rate_col[group_mask].round(2).tolist()
    sub_names = df.loc[df_reg.index[group_mask], "settlement_name"].tolist() if "settlement_name" in df.columns else []
    sub_ses = ses_cluster_col[group_mask].fillna(5).astype(int).tolist()
    # OLS trend line endpoints
    xv = np.array(sub_child, dtype=float)
    yv = np.array(sub_adult, dtype=float)
    valid = ~(np.isnan(xv) | np.isnan(yv))
    if valid.sum() >= 4:
        slope, intercept, *_ = _linregress(xv[valid], yv[valid])
        x_min, x_max = float(xv[valid].min()), float(xv[valid].max())
        trend = {"x0": round(x_min, 2), "y0": round(slope * x_min + intercept, 2),
                 "x1": round(x_max, 2), "y1": round(slope * x_max + intercept, 2),
                 "slope": round(slope, 3)}
    else:
        trend = None
    inter_scatter_data["groups"].append({
        "name": group_label,
        "child_rate": sub_child,
        "adult_rate": sub_adult,
        "names": sub_names,
        "ses_cluster": sub_ses,
        "color": color,
        "r": grp_r, "n": grp_n,
        "trend": trend,
    })

print(f"  Overall r={r_overall}, rho={rho_overall} (N={n_inter})")
print(f"  Peripheral r={r_periph} (N={n_periph}), Non-peripheral r={r_nonperiph} (N={n_nonperiph})")
print(f"  Fisher Z={fisher_z}, p={fisher_p}")
print(f"  High-Risk Cluster: {n_high_risk} settlements ({high_risk_pct}%)")


# ── HTML generation ──────────────────────────────────────────────────────────

print("Generating HTML ...")

# Build ensemble underutilization table rows
ensemble_pop = pd.to_numeric(
    df.loc[df_reg.index, "total_population"], errors="coerce"
).values
ensemble_table_rows = ""
for rank, idx in enumerate(under_candidates.index):
    pos = list(df_reg.index).index(idx)
    name = under_candidates.loc[idx, "settlement_name"]
    actual = under_candidates.loc[idx, "actual_rate"]
    predicted = under_candidates.loc[idx, "mean_pred_rate"]
    gap = under_candidates.loc[idx, "gap_pp"]
    pop = int(ensemble_pop[pos]) if not np.isnan(ensemble_pop[pos]) else 0
    ensemble_table_rows += f"""<tr>
      <td>{rank + 1}</td>
      <td>{html_lib.escape(str(name))}</td>
      <td>{actual:.2f}%</td>
      <td>{predicted:.2f}%</td>
      <td style="color:#55A868;font-weight:600;">{gap:+.2f}</td>
      <td>{pop:,}</td>
    </tr>"""

# Build OLS underutilization table rows
ols_table_rows = ""
for i, (name, actual, predicted, residual, pop) in enumerate(top_under_table):
    ols_table_rows += f"""<tr>
      <td>{i + 1}</td>
      <td>{html_lib.escape(str(name))}</td>
      <td>{actual:.2f}%</td>
      <td>{predicted:.2f}%</td>
      <td style="color:var(--negative);font-weight:600;">{residual:+.2f}</td>
      <td>{pop:,}</td>
    </tr>"""

# Build OLS prediction lookup for explorer table
ols_pred_map = dict(zip(df_ols_clean.index, y_ols_pred))
ols_actual_map = dict(zip(df_ols_clean.index, y_ols))
ols_resid_map = dict(zip(df_ols_clean.index, ols_residuals))

# Build full settlement explorer table (all settlements, sorted by gap)
all_settlements_rows = ""
df_gap_sorted = df_gap.sort_values("gap_pp", ascending=False)
for rank, (idx, row) in enumerate(df_gap_sorted.iterrows()):
    pos = list(df_reg.index).index(idx)
    name = row.get("settlement_name", "")
    actual = row["actual_rate"]
    predicted = row["mean_pred_rate"]
    gap = row["gap_pp"]
    pop = int(ensemble_pop[pos]) if not np.isnan(ensemble_pop[pos]) else 0
    seg = row["segment"]
    arab = row.get("arab_population_percentage", 0)
    haredi_v = pd.to_numeric(df.loc[idx, "haredi_population_percentage"], errors="coerce") if "haredi_population_percentage" in df.columns else 0
    if pd.isna(arab): arab = 0
    if pd.isna(haredi_v): haredi_v = 0
    group = "Arab" if arab > 50 else ("Haredi" if haredi_v > 50 else "Other")
    gap_color = "#55A868" if gap > 0 else "#C44E52"
    seg_badge = ""
    if seg == "Potential Under-utilization":
        seg_badge = '<span style="background:#55A868;color:#fff;padding:1px 6px;border-radius:3px;font-size:0.75rem;">Under-util</span>'
    elif seg == "Hidden Burden":
        seg_badge = '<span style="background:#C44E52;color:#fff;padding:1px 6px;border-radius:3px;font-size:0.75rem;">Hidden</span>'
    # OLS status for this settlement
    ols_seg = ols_status_map.get(idx, "")
    ols_badge = ""
    if ols_seg == "Underutilization":
        ols_badge = '<span style="background:#55A868;color:#fff;padding:1px 6px;border-radius:3px;font-size:0.75rem;">Under-util</span>'
    elif ols_seg == "Dependency":
        ols_badge = '<span style="background:#C44E52;color:#fff;padding:1px 6px;border-radius:3px;font-size:0.75rem;">Hidden</span>'
    # OLS predicted values for this settlement
    ols_pred_v = ols_pred_map.get(idx, float("nan"))
    ols_actual_v = ols_actual_map.get(idx, float("nan"))
    ols_resid_v = ols_resid_map.get(idx, float("nan"))
    has_ols = not np.isnan(ols_pred_v)
    ols_gap_v = -ols_resid_v if has_ols else float("nan")  # positive = under-util (predicted > actual)
    ols_pred_str = f"{ols_pred_v:.1f}" if has_ols else ""
    ols_gap_str = f"{ols_gap_v:+.1f}" if has_ols else ""
    all_settlements_rows += f"""<tr data-ens-pred="{predicted:.1f}" data-ens-gap="{gap:+.1f}" data-ols-pred="{ols_pred_str}" data-ols-gap="{ols_gap_str}">
      <td>{html_lib.escape(str(name))}</td>
      <td>{actual:.1f}%</td>
      <td class="col-pred">{predicted:.1f}%</td>
      <td class="col-gap" style="color:{gap_color};font-weight:600;">{gap:+.1f}</td>
      <td>{pop:,}</td>
      <td>{group}</td>
      <td>{seg_badge}</td>
      <td>{ols_badge}</td>
    </tr>"""

# ── Assemble the full HTML ──────────────────────────────────────────────────

html_output = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Where Need Meets Silence</title>
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
      --text-secondary: #495057;
      --text-on-dark: #ffffff;
      --color-1: #4C72B0;
      --color-2: #DD8452;
      --color-3: #55A868;
      --color-4: #C44E52;
      --color-5: #8172B3;
      --color-6: #937860;
      --positive: #28a745;
      --negative: #dc3545;
      --neutral: #4C72B0;
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
      height: 52px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }}
    .top-nav .brand {{
      font-weight: 700;
      font-size: 0.9rem;
      margin-right: 28px;
      white-space: nowrap;
    }}
    .top-nav .nav-links {{
      display: flex;
      gap: 2px;
      overflow-x: auto;
      scrollbar-width: thin;
    }}
    .top-nav .nav-links::-webkit-scrollbar {{ height: 3px; }}
    .top-nav .nav-links::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.3); border-radius: 2px; }}
    .top-nav .nav-links a {{
      color: rgba(255,255,255,0.7);
      text-decoration: none;
      padding: 5px 12px;
      border-radius: 6px;
      font-size: 0.78rem;
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
      margin-bottom: 64px;
      scroll-margin-top: 68px;
      border-top: 1px solid #dee2e6;
      padding-top: 32px;
    }}
    .section:first-child {{
      border-top: none;
      padding-top: 0;
    }}

    .section-title {{
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--text-primary);
      margin-bottom: 6px;
    }}

    .section-subtitle {{
      font-size: 1.05rem;
      color: var(--text-secondary);
      margin-bottom: 20px;
      max-width: 720px;
      line-height: 1.5;
    }}

    /* ── Q-badges ── */
    .q-badge {{
      display: inline-block;
      background: var(--color-1);
      color: #fff;
      font-size: 0.72rem;
      font-weight: 700;
      padding: 2px 10px;
      border-radius: 12px;
      margin-right: 8px;
      vertical-align: middle;
      letter-spacing: 0.03em;
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
    .kpi-card.hero .kpi-value {{
      font-size: 2.8rem;
    }}
    .kpi-card.secondary .kpi-value {{
      font-size: 1.8rem;
      color: var(--text-secondary);
    }}
    .kpi-card .kpi-value.problem {{ color: var(--negative); }}
    .kpi-card .kpi-value.solution {{ color: var(--positive); }}
    .kpi-card .kpi-value.neutral {{ color: var(--neutral); }}
    .kpi-card .kpi-label {{
      font-size: 0.82rem;
      color: var(--text-secondary);
      margin-top: 6px;
      font-weight: 600;
    }}
    .kpi-context {{
      font-size: 0.78rem;
      color: var(--text-secondary);
      margin-top: 4px;
      line-height: 1.3;
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
      padding-bottom: 8px;
      border-bottom: 1px solid #f1f3f5;
      color: var(--text-primary);
    }}
    .card p {{
      font-size: 0.9rem;
      color: var(--text-secondary);
      line-height: 1.7;
      max-width: 600px;
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
      margin-bottom: 24px;
    }}
    .chart-card .chart-title {{
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-primary);
      margin-bottom: 4px;
    }}
    .chart-annotation {{
      font-size: 0.82rem;
      font-style: italic;
      color: var(--text-secondary);
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
    .chart-card .chart-wrap.hbar {{
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

    /* ── TL;DR box ── */
    .tldr-box {{
      background: #fffbeb;
      border: 2px solid #f59e0b;
      border-radius: var(--radius);
      padding: 20px 24px;
      margin-bottom: 24px;
    }}
    .tldr-box .tldr-label {{
      font-size: 0.78rem;
      font-weight: 700;
      color: #92400e;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }}
    .tldr-box p {{
      font-size: 0.95rem;
      color: #78350f;
      line-height: 1.6;
      max-width: 720px;
    }}
    .tldr-box strong {{ color: #451a03; }}

    /* ── Pull quote ── */
    .pull-quote {{
      font-size: 1.3rem;
      font-weight: 700;
      color: var(--negative);
      line-height: 1.4;
      margin: 16px 0;
      padding: 16px 24px;
      border-left: 4px solid var(--negative);
      background: #fff5f5;
      border-radius: 0 var(--radius) var(--radius) 0;
    }}
    .pull-quote .pq-source {{
      font-size: 0.82rem;
      font-weight: 500;
      color: var(--text-secondary);
      margin-top: 8px;
    }}

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
      margin-bottom: 12px;
    }}
    .header-banner .subtitle {{
      font-size: 1.05rem;
      color: rgba(255,255,255,0.75);
      margin-bottom: 16px;
      max-width: 720px;
      margin-left: auto;
      margin-right: auto;
      line-height: 1.5;
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
      .chart-card .chart-wrap.hbar {{ height: 320px; }}
      .pull-quote {{ font-size: 1.1rem; }}
    }}

    /* ── Print ── */
    @media print {{
      .top-nav {{ display: none; }}
      body {{ background: #fff; }}
      .section {{ break-inside: avoid; page-break-inside: avoid; }}
      .chart-card {{ break-inside: avoid; page-break-inside: avoid; }}
      .chart-card, .card, .kpi-card {{ box-shadow: none; border: 1px solid #ccc; }}
      .chart-card .chart-wrap, .chart-card .chart-wrap.tall, .chart-card .chart-wrap.hbar {{ height: 260px; }}
      .big-number-card .big-val {{ font-size: 2.5rem; }}
    }}

    /* ── Reduced motion ── */
    @media (prefers-reduced-motion: reduce) {{
      *, *::before, *::after {{
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }}
    }}
  </style>
</head>
<body>

<!-- Navigation -->
<nav class="top-nav">
  <div class="brand">Where Need Meets Silence</div>
  <div class="nav-links">
    <a href="#problem" class="active">The Problem</a>
    <a href="#missing">Missing Claimants</a>
    <a href="#childhood">Childhood Cycle</a>
    <a href="#wall">The Wall</a>
    <a href="#ruled-out">Ruled Out</a>
    <a href="#drivers">What Drives It</a>
    <a href="#action">Action Plan</a>
  </div>
</nav>

<!-- Header Banner -->
<header class="header-banner">
  <h1>Where Need Meets Silence</h1>
  <p class="subtitle">Why do some Israeli communities receive far fewer disability benefits than their conditions would predict?</p>
  <p class="meta">TovTech Research Group &nbsp;|&nbsp; National Insurance Institute data, Dec 2024 &nbsp;|&nbsp; 278 settlements &nbsp;|&nbsp; 4 research questions</p>
</header>

<div class="dashboard">

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 1: The Problem -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="problem" style="border-top:none;padding-top:0;">
  <h2 class="section-title">People Who Qualify for Disability Benefits Are Not Receiving Them</h2>
  <p class="section-subtitle">Israel&rsquo;s National Insurance Institute pays disability benefits to hundreds of thousands of people.
  But in some settlements, far fewer people claim than we would expect given their socio-economic conditions.
  We analyzed {n_reg} settlements with {n_features} indicators to find out where &mdash; and why.</p>

  <div class="tldr-box">
    <div class="tldr-label">What we did</div>
    <p>We trained machine learning models to predict disability benefit rates from socio-economic data
    (income, education, demographics, geography). Settlements where <strong>actual claiming is far below
    the prediction</strong> are potential under-utilization pockets &mdash; places where eligible people
    may face barriers to access.</p>
  </div>

  <div class="kpi-row">
    <div class="kpi-card hero">
      <div class="kpi-value neutral">{n_reg}</div>
      <div class="kpi-label">Settlements Analyzed</div>
      <div class="kpi-context">Across all regions and population groups</div>
    </div>
    <div class="kpi-card hero">
      <div class="kpi-value neutral">{n_features}</div>
      <div class="kpi-label">Socio-Economic Indicators</div>
      <div class="kpi-context">Income, education, demographics, geography</div>
    </div>
    <div class="kpi-card secondary">
      <div class="kpi-value neutral">4</div>
      <div class="kpi-label">Research Questions</div>
      <div class="kpi-context">Under-utilization, intergenerational, conflict, distance</div>
    </div>
    <div class="kpi-card secondary">
      <div class="kpi-value neutral">4</div>
      <div class="kpi-label">ML Models</div>
      <div class="kpi-context">Random Forest, XGBoost, TabPFN, Linear Regression</div>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 2: Missing Claimants (Q1) -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="missing">
  <h2 class="section-title"><span class="q-badge">Research Question 1</span> Settlements That Claim Far Less Than Their Conditions Predict</h2>
  <p class="section-subtitle">We used two methods &mdash; a non-linear ensemble and a linear regression &mdash; to predict disability rates from socio-economic data. Both use &plusmn;1.5&sigma; residuals to flag anomalies. Each method reveals a different blind spot.</p>

  <!-- ── Ensemble ── -->
  <h3 style="font-size:1.05rem;font-weight:600;margin:0 0 12px;color:var(--text-primary);">Ensemble Model (RF + XGBoost + TabPFN) &mdash; 15 features</h3>
  <div class="chart-card">
    <div class="chart-title">Expected vs Observed Disability Rate</div>
    <div class="chart-annotation"><strong style="color:#55A868;">Green diamonds</strong> = under-utilization. <strong style="color:#C44E52;">Red crosses</strong> = hidden burden. Dashed bands = &plusmn;1.5&sigma; ({round(gap_threshold, 1)} pp).</div>
    <div style="margin:8px 0 4px;display:flex;align-items:center;gap:8px;">
      <input id="search_ensemble" type="text" placeholder="Search settlement..." style="padding:5px 10px;border:1px solid #ddd;border-radius:6px;font-size:0.85rem;width:220px;outline:none;" />
      <span id="search_ensemble_result" style="font-size:0.8rem;color:var(--text-secondary);"></span>
    </div>
    <div class="chart-wrap tall"><canvas id="chart_gap_scatter"></canvas></div>
  </div>

  <div class="kpi-row">
    <div class="kpi-card hero">
      <div class="kpi-value" style="color:#55A868;">{n_under}</div>
      <div class="kpi-label">Under-Utilization</div>
      <div class="kpi-context">Actual &lt; expected by &ge; {round(gap_threshold, 1)} pp</div>
    </div>
    <div class="kpi-card hero">
      <div class="kpi-value problem">{n_hidden}</div>
      <div class="kpi-label">Hidden Burden</div>
      <div class="kpi-context">Actual &gt; expected by &ge; {round(gap_threshold, 1)} pp</div>
    </div>
  </div>

  <div class="card-grid" style="grid-template-columns: repeat(3, 1fr);">
    <div class="card" style="border-top:3px solid #6366F1;">
      <h3>Haredi &mdash; {under_haredi_pct}%</h3>
      <p><strong>{n_haredi_under} of {n_under}</strong> under-utilization settlements are Haredi-majority (&gt;50%).</p>
    </div>
    <div class="card" style="border-top:3px solid #F59E0B;">
      <h3>Arab / Druze &mdash; {under_arab_pct}%</h3>
      <p><strong>{n_arab_under} of {n_under}</strong> under-utilization settlements are Arab-majority (&gt;50%).</p>
    </div>
    <div class="card" style="border-top:3px solid #94A3B8;">
      <h3>Other &mdash; {under_other_pct}%</h3>
      <p><strong>{n_other_under} of {n_under}</strong> are smaller secular/mixed settlements.</p>
    </div>
  </div>

  <!-- ── Linear Regression ── -->
  <h3 style="font-size:1.05rem;font-weight:600;margin:28px 0 12px;color:var(--text-primary);">Linear Regression &mdash; 8 features, R&sup2; = {ols_r2_full}</h3>
  <div class="chart-card">
    <div class="chart-title">Expected vs Observed Disability Rate</div>
    <div class="chart-annotation">Threshold &plusmn;1.5&sigma; flags <strong>{n_ols_under}</strong> under-utilization and <strong>{n_ols_dependency}</strong> hidden burden anomalies.</div>
    <div style="margin:8px 0 4px;display:flex;align-items:center;gap:8px;">
      <input id="search_ols" type="text" placeholder="Search settlement..." style="padding:5px 10px;border:1px solid #ddd;border-radius:6px;font-size:0.85rem;width:220px;outline:none;" />
      <span id="search_ols_result" style="font-size:0.8rem;color:var(--text-secondary);"></span>
    </div>
    <div class="chart-wrap tall"><canvas id="chart_anomaly_scatter"></canvas></div>
  </div>

  <div class="kpi-row">
    <div class="kpi-card hero">
      <div class="kpi-value" style="color:#55A868;">{n_ols_under}</div>
      <div class="kpi-label">Under-Utilization</div>
      <div class="kpi-context">Actual &lt; expected by &ge; {round(ols_threshold, 1)} pp</div>
    </div>
    <div class="kpi-card hero">
      <div class="kpi-value problem">{n_ols_dependency}</div>
      <div class="kpi-label">Hidden Burden</div>
      <div class="kpi-context">Actual &gt; expected by &ge; {round(ols_threshold, 1)} pp</div>
    </div>
  </div>

  <div class="card-grid" style="grid-template-columns: repeat(3, 1fr);">
    <div class="card" style="border-top:3px solid #6366F1;">
      <h3>Haredi &mdash; {ols_under_haredi_pct}%</h3>
      <p><strong>{n_ols_haredi_under} of {n_ols_under}</strong> under-utilization settlements are Haredi-majority (&gt;50%).</p>
    </div>
    <div class="card" style="border-top:3px solid #F59E0B;">
      <h3>Arab / Druze &mdash; {ols_under_arab_pct}%</h3>
      <p><strong>{n_ols_arab_under} of {n_ols_under}</strong> under-utilization settlements are Arab-majority (&gt;50%).</p>
    </div>
    <div class="card" style="border-top:3px solid #94A3B8;">
      <h3>Other &mdash; {ols_under_other_pct}%</h3>
      <p><strong>{n_ols_other_under} of {n_ols_under}</strong> are other secular/mixed settlements.</p>
    </div>
  </div>

  <div class="insight-callout">
    <strong>Two methods, different blind spots:</strong> The ensemble (15 features, non-linear) flags mostly Haredi and mixed settlements. The linear regression (8 features) picks up more Arab settlements. Together they cover a wider range of potential under-utilization than either method alone.
  </div>

  <h3 style="font-size:1.1rem;font-weight:600;margin:28px 0 12px;">Explore All {n_reg} Settlements</h3>
  <div style="margin-bottom:12px;display:flex;flex-wrap:wrap;align-items:center;gap:10px;">
    <input id="table_search" type="text" placeholder="&#128269; Search settlement name..." style="padding:8px 14px;border:2px solid #cbd5e1;border-radius:8px;font-size:0.95rem;width:260px;outline:none;background:#f8fafc;" />
    <select id="table_filter_group" style="padding:8px 12px;border:2px solid #cbd5e1;border-radius:8px;font-size:0.9rem;background:#f8fafc;cursor:pointer;">
      <option value="">All groups</option>
      <option value="Arab">Arab (&gt;50%)</option>
      <option value="Haredi">Haredi (&gt;50%)</option>
      <option value="Other">Other</option>
    </select>
    <select id="table_filter_status" style="padding:8px 12px;border:2px solid #cbd5e1;border-radius:8px;font-size:0.9rem;background:#f8fafc;cursor:pointer;">
      <option value="">All statuses</option>
      <option value="ens-under">Ensemble: Under-utilization</option>
      <option value="ens-hidden">Ensemble: Hidden burden</option>
      <option value="ols-under">OLS: Under-utilization</option>
      <option value="ols-hidden">OLS: Hidden burden</option>
      <option value="any-flag">Any flag (either method)</option>
    </select>
    <span id="table_count" style="font-size:0.85rem;color:var(--text-secondary);"></span>
  </div>
  <div class="data-table-wrap" style="max-height:420px;overflow-y:auto;">
    <table class="data-table" id="explorer_table">
      <thead style="position:sticky;top:0;background:#fff;z-index:1;">
        <tr>
          <th onclick="sortTable('explorer_table', 0)">Settlement <span class="sort-arrow">&#9650;&#9660;</span></th>
          <th onclick="sortTable('explorer_table', 1)">Actual <span class="sort-arrow">&#9650;&#9660;</span></th>
          <th onclick="sortTable('explorer_table', 2)" id="th_pred">Expected (Ensemble) <span class="sort-arrow">&#9650;&#9660;</span></th>
          <th onclick="sortTable('explorer_table', 3)" id="th_gap">Gap (Ensemble) <span class="sort-arrow">&#9650;&#9660;</span></th>
          <th onclick="sortTable('explorer_table', 4)">Population <span class="sort-arrow">&#9650;&#9660;</span></th>
          <th onclick="sortTable('explorer_table', 5)">Group <span class="sort-arrow">&#9650;&#9660;</span></th>
          <th>Ensemble</th>
          <th>OLS</th>
        </tr>
      </thead>
      <tbody>
        {all_settlements_rows}
      </tbody>
    </table>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 3: Childhood Cycle (Q2) -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="childhood">
  <h2 class="section-title"><span class="q-badge">Research Question 2</span> Where Children Need Help, Adults Need It Too &mdash; Even in Wealthy Areas</h2>
  <p class="section-subtitle">Child disability rates correlate with adult disability rates across all regions. The surprise: the correlation is <strong>stronger</strong> in non-peripheral settlements (Fisher&rsquo;s Z {fisher_sig}).</p>

  <!-- KPI row: Overall + Peripheral vs Non-Peripheral -->
  <div class="kpi-row">
    <div class="kpi-card hero">
      <div class="kpi-value neutral">{r_overall}</div>
      <div class="kpi-label">Overall Pearson r</div>
      <div class="kpi-context">&rho; = {rho_overall}, N = {n_inter} settlements</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value" style="color:var(--color-2);">{r_periph}</div>
      <div class="kpi-label">Peripheral r (1-5, N={n_periph})</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value neutral">{r_nonperiph}</div>
      <div class="kpi-label">Non-Peripheral r (6-10, N={n_nonperiph})</div>
      <div class="kpi-context">Fisher Z = {fisher_z}, {fisher_sig}</div>
    </div>
  </div>

  <!-- Dual-panel scatter: Peripheral vs Non-Peripheral -->
  <div class="chart-card">
    <div class="chart-title">Child Disability Rate vs Adult Disability Rate &mdash; Peripheral vs Non-Peripheral</div>
    <div class="chart-annotation">Each dot is a settlement, colored by SES cluster (red=low, green=high). Dashed line = OLS trend. The steeper slope and tighter fit in non-peripheral settlements (r&nbsp;=&nbsp;{r_nonperiph}) confirms that the intergenerational pattern is <strong>not</strong> confined to the periphery.</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
      <div class="chart-wrap tall"><canvas id="chart_inter_periph"></canvas></div>
      <div class="chart-wrap tall"><canvas id="chart_inter_nonperiph"></canvas></div>
    </div>
  </div>

  <!-- High-Risk Cluster card -->
  <div class="big-number-card" style="border-left:4px solid var(--negative);">
    <div class="big-val">{n_high_risk}</div>
    <div class="big-desc">settlements ({high_risk_pct}%) where <strong>both</strong> adult and child disability rates exceed the 75th percentile &mdash; potential intergenerational disability traps</div>
  </div>

  <!-- Boxplot: High-Risk vs Rest -->
  <div class="chart-card">
    <div class="chart-title">High-Risk Cluster vs Rest: Socio-Economic Profile</div>
    <div class="chart-annotation">{n_high_risk} settlements where both adult &amp; child disability &gt; P75, compared against the remaining {n_inter - n_high_risk}. Median salary: &#8362;{sal_hr:,.0f} vs &#8362;{sal_rest:,.0f}.</div>
    <div class="chart-wrap tall"><canvas id="chart_hr_boxplot"></canvas></div>
  </div>

  <!-- Welch's T-test comparison cards -->
  <h3 style="font-size:1.1rem;font-weight:600;margin:28px 0 8px;">Statistical Comparison: High-Risk vs Rest (Welch&rsquo;s T-test)</h3>
  <p style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:16px;">
    Mean values compared between the <strong>{n_high_risk} High-Risk</strong> settlements and the remaining <strong>{n_inter - n_high_risk}</strong>.
    Only indicators with <strong>p &lt; 0.05</strong> are statistically significant.
  </p>
  <div class="card-grid" style="grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));">
    {"".join(f'''
    <div class="card" style="border-top:3px solid {'var(--negative)' if c['significant'] else 'var(--border)'};">
      <h3 style="font-size:0.95rem;margin-bottom:8px;">{c['label']}</h3>
      <p style="font-size:0.78rem;color:var(--text-secondary);margin-bottom:10px;">{c['unit']}</p>
      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
        <div style="text-align:center;">
          <div style="font-size:1.3rem;font-weight:700;color:var(--negative);">{c['hr_mean']:.1f}</div>
          <div style="font-size:0.72rem;color:var(--text-secondary);">High-Risk</div>
        </div>
        <div style="text-align:center;font-size:1.1rem;color:var(--text-secondary);align-self:center;">vs</div>
        <div style="text-align:center;">
          <div style="font-size:1.3rem;font-weight:700;color:var(--color-1);">{c['rest_mean']:.1f}</div>
          <div style="font-size:0.72rem;color:var(--text-secondary);">Rest</div>
        </div>
      </div>
      <div style="font-size:0.82rem;padding:6px 8px;border-radius:6px;background:{'rgba(196,78,82,0.1)' if c['significant'] else 'rgba(128,128,128,0.08)'};">
        <strong>&Delta; = {c['delta']:+.2f}</strong> &nbsp;|&nbsp;
        t = {c['t_stat']:.2f} &nbsp;|&nbsp;
        <span style="color:{'var(--negative)' if c['significant'] else 'var(--text-secondary)'};">
          p = {c['p_val']:.4f} {'&#x2713;' if c['significant'] else '(n.s.)'}
        </span>
      </div>
    </div>''' for c in hr_comparison)}
  </div>

  <!-- Insight cards -->
  <div class="card-grid" style="margin-top:var(--gap);">
    <div class="card" style="border-top:3px solid var(--negative);">
      <h3>The Pattern That Shouldn&rsquo;t Exist</h3>
      <p>In non-peripheral settlements (r = {r_nonperiph}), the child-adult disability link is
      <strong>stronger</strong> than in the periphery (r = {r_periph}).
      The difference is statistically significant (Fisher&rsquo;s Z = {fisher_z}, {fisher_sig}).
      Wealth alone does not break the cycle.</p>
    </div>
    <div class="card" style="border-top:3px solid var(--color-1);">
      <h3>Overall Picture</h3>
      <p>High-Risk settlements combine: low socio-economic standing ({ses_hr:.2f} vs {ses_rest:.2f}),
      lower wages (&#8362;{sal_hr:,.0f} vs &#8362;{sal_rest:,.0f}), higher income support dependence,
      and lower education &mdash; creating a <strong>self-reinforcing environment</strong> where both
      adults and children depend on disability benefits.
      {n_significant} of {len(hr_comparison)} indicators show statistically significant differences (Welch&rsquo;s T-test).</p>
    </div>
    <div class="card" style="border-top:3px solid var(--color-2);">
      <h3>What This Means for Policy</h3>
      <p>The {n_high_risk} high-risk settlements need holistic <strong>family-level</strong> rehabilitation programs,
      not individual disability support alone. Breaking the intergenerational cycle requires
      simultaneous intervention in education, employment, and healthcare access.</p>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 4: The Wall (CLIMAX — Arab R2 + integrated evidence) -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="wall">
  <h2 class="section-title">Every Model We Tried Hit the Same Wall</h2>
  <p class="section-subtitle">We tested Random Forest, XGBoost, TabPFN, and Linear Regression. All four models explain disability patterns well for most settlements &mdash; and fail for Arab communities.</p>

  <div class="big-number-card">
    <div class="big-val">~{arab_unexplained_pct}%</div>
    <div class="big-desc">of the variance in Arab disability claiming <strong>cannot be explained</strong>
    by any combination of {n_features} socio-economic indicators</div>
  </div>

  <div class="chart-card">
    <div class="chart-title">Model R&sup2; by Population Sector</div>
    <div class="chart-annotation">R&sup2; = {arab_best_r2:.3f} for Arab settlements vs {fmt_num(best_secular_r2, 3)} for Secular. The wall is consistent across all model architectures.</div>
    <div class="chart-wrap"><canvas id="chart_arab_r2"></canvas></div>
  </div>

  <div class="pull-quote">
    &ldquo;It&rsquo;s Disgraceful Going through All this for Being an Arab and Disabled&rdquo;
    <div class="pq-source">Alhuzeel et al. (2023) &mdash; <em>Scandinavian Journal of Disability Research</em>. Interviews with 15 Arab Israelis revealed multi-level barriers: language (Hebrew-only forms and committees), excessive bureaucracy (3 medical committees vs 1), lack of information about rights in Arabic, and cultural stigma within the community itself.</div>
  </div>

  <div class="pull-quote" style="border-left-color:var(--color-1);background:#eff6ff;color:var(--color-1);">
    Arab self-reported disability prevalence: <strong>21%</strong> &mdash; Jewish: <strong>19%</strong>.
    Yet benefit claiming is <em>lower</em>. 35% of approved beneficiaries never exercised eligibility.
    <div class="pq-source" style="color:var(--text-secondary);">Brookdale Institute (2024) &mdash; People with Disabilities in the Arab Population</div>
  </div>

  <div class="card-grid" style="margin-top:var(--gap);grid-template-columns:1fr;">
    <div class="card" style="border-top:3px solid var(--negative);">
      <h3>We Removed the Arab Variable &mdash; Nothing Changed</h3>
      <p>We re-ran the ensemble without <code>arab_population_percentage</code>.
      The under-utilization list stayed <strong>identical</strong>.
      The model doesn&rsquo;t need an &ldquo;Arab&rdquo; label &mdash; it reconstructs the group from
      the remaining 14 indicators: low SES, low salary, zero Haredi&nbsp;%, specific peripherality
      and education patterns. No single variable carries the signal; <strong>it is distributed
      across the entire socio-economic profile</strong>.</p>
      <p style="margin-top:8px;">This means the model <em>normalizes</em> the under-claiming pattern.
      It learns &ldquo;settlements that look like this tend to claim less&rdquo; and adjusts its
      prediction downward &mdash; turning systemic access barriers into a statistical baseline.
      The discrimination is not in one column; it is woven into the structure of the data.</p>
    </div>
  </div>

  <div class="card-grid" style="margin-top:var(--gap);">
    <div class="card" style="border-top:3px solid var(--color-5);">
      <h3>This Has Happened Before</h3>
      <p><strong>Obermeyer et al. (2019, Science)</strong>: A US healthcare algorithm trained on
         200 million patients used <em>cost</em> as proxy for <em>need</em>. Because the system
         spent less on Black patients with equivalent illness, the algorithm concluded they were healthier.</p>
      <p style="margin-top:8px;">Our case is analogous: NII data records <em>benefit receipt</em> as
         proxy for <em>disability prevalence</em>. In communities facing access barriers, fewer people
         claim &mdash; creating the illusion of lower need.</p>
    </div>
    <div class="card" style="border-top:3px solid var(--positive);">
      <h3>Convergence of Evidence</h3>
      <p>Our data, the qualitative research, and the international precedent all point to the same conclusion:</p>
      <ul>
        <li>The <strong>R&sup2; wall</strong> (~{arab_unexplained_pct}% unexplained) = unmeasured barriers</li>
        <li>The <strong>under-utilization clusters</strong> ({under_arab_pct}% Arab-majority) = families not claiming despite need</li>
        <li>The <strong>hidden burden clusters</strong> ({hidden_arab_pct}% Arab-majority) = mostly non-Arab; the Arab gap is about <em>access</em>, not <em>over-diagnosis</em></li>
        <li>The <strong>intergenerational pattern</strong> = access failure starts in childhood</li>
      </ul>
    </div>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 5: Ruled Out (Q3 + Q4 combined) -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="ruled-out">
  <h2 class="section-title">It's Not the War. It's Not the Distance.</h2>
  <p class="section-subtitle">Two alternative explanations tested and rejected: the Swords of Iron conflict and physical distance to NII branches.</p>

  <!-- Subsection A: Q3 Swords of Iron -->
  <h3 style="font-size:1.15rem;font-weight:600;margin:0 0 12px;"><span class="q-badge">Research Question 3</span> Did the Conflict Create a Disability Crisis?</h3>

  <div class="kpi-row">
    <div class="kpi-card">
      <div class="kpi-value problem">{front_delta_total:+.2f} pp</div>
      <div class="kpi-label">Frontline Change (2023&rarr;2025)</div>
      <div class="kpi-context">{front_rate_2023}% &rarr; {front_rate_2025}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value neutral">{nf_delta_total:+.2f} pp</div>
      <div class="kpi-label">Non-Frontline Change (2023&rarr;2025)</div>
      <div class="kpi-context">{nf_rate_2023}% &rarr; {nf_rate_2025}%</div>
    </div>
  </div>

  <div class="chart-card">
    <div class="chart-title">Disability Benefit Rate: Frontline ({n_frontline_panel} settlements) vs Non-Frontline ({n_nonfrontline_panel} settlements)</div>
    <div class="chart-annotation">Rates rose everywhere, not just in conflict zones. The trend is structural, not crisis-driven.</div>
    <div class="chart-wrap"><canvas id="chart_temporal"></canvas></div>
  </div>

  <div class="insight-callout">
    <strong>Conclusion:</strong> Disability rates rose <strong>everywhere</strong>, not just in
    frontline zones. The conflict did not create a new disability spike &mdash;
    it amplified existing vulnerabilities.
  </div>

  <!-- Subsection B: Q4 Distance -->
  <h3 style="font-size:1.15rem;font-weight:600;margin:24px 0 12px;"><span class="q-badge">Research Question 4</span> Is Physical Distance the Barrier?</h3>

  <div class="chart-card">
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:4px;flex-wrap:wrap;">
      <div class="chart-title" id="dist_chart_title" style="margin-bottom:0;">Disability Rate vs Distance to Nearest NII Branch (km)</div>
      <select id="dist_type_select" onchange="updateDistanceChart()" style="padding:6px 12px;border-radius:6px;border:1px solid #dee2e6;font-size:0.85rem;background:#fff;cursor:pointer;">
        <option value="dist_any">Any BTL Branch</option>
        <option value="dist_central">Central Branch</option>
      </select>
    </div>
    <div class="chart-annotation">Each dot is a settlement. No visible trend &mdash; distance does not predict disability claiming.</div>
    <div class="chart-wrap tall"><canvas id="chart_distance_scatter"></canvas></div>
  </div>

  <div class="chart-card">
    <div class="chart-title">Same Scale Comparison: SES (slope {trend_data['slope_ses']:+.2f}) vs Distance (slope {trend_data['slope_dist']:+.2f})</div>
    <div class="chart-annotation">SES is a strong predictor; distance is nearly flat.</div>
    <div class="chart-wrap"><canvas id="chart_trend_lines"></canvas></div>
  </div>

  <div class="insight-callout">
    <strong>Conclusion:</strong> Physical distance to NII branches explains almost nothing.
    The barrier is not geographic &mdash; it is informational, linguistic, cultural, and institutional.
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 6: What Drives Disability -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="drivers">
  <h2 class="section-title">Income Support Is the Strongest Driver of Disability Rates</h2>
  <p class="section-subtitle">Knowing what drives disability tells us what to target. Two complementary views: Linear Regression shows direction (increases vs decreases), tree models show magnitude.</p>

  <div class="card-grid">
    <div class="chart-card">
      <div class="chart-title">Linear Regression: Which Factors Increase or Decrease Disability?</div>
      <div class="chart-annotation"><strong style="color:#C44E52;">Red bars</strong> = factor increases disability rate. <strong style="color:#4C72B0;">Blue bars</strong> = factor decreases it. Income support is the strongest positive driver; salary is the strongest negative. R&sup2; = {ols_cv_r2}.</div>
      <div class="chart-wrap hbar"><canvas id="chart_ols_coeff"></canvas></div>
    </div>
    <div class="chart-card">
      <div class="chart-title">Tree Models: Which Factors Matter Most?</div>
      <div class="chart-annotation">Importance regardless of direction. Averaged across Random Forest + XGBoost. Best ensemble R&sup2; = {best_r2_val}.</div>
      <div class="chart-wrap hbar"><canvas id="chart_importance"></canvas></div>
    </div>
  </div>

  <div class="insight-callout">
    <strong>Both methods agree:</strong> income support rate, salary, and Arab population percentage are the top factors.
    Linear Regression tells us the <em>direction</em> (what to increase, what to decrease).
    Tree models tell us the <em>importance</em> (where to focus resources first).
    Together, they point directly to the actions in the next section.
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════════ -->
<!-- Section 7: Action Plan -->
<!-- ════════════════════════════════════════════════════════════════════════ -->
<div class="section" id="action">
  <h2 class="section-title">Five Actions to Close the Gap</h2>
  <p class="section-subtitle">Each recommendation is tied to specific findings from this research.</p>

  <div class="rec-grid">
    <div class="rec-card">
      <div class="rec-num">1</div>
      <h4>Targeted Outreach in {n_under} Flagged Settlements</h4>
      <p><strong>From Q1:</strong> the {n_under} flagged settlements should receive proactive information
      campaigns &mdash; in Arabic, with community-based intermediaries and local
      advocacy organizations. {under_arab_pct}% are Arab-majority.</p>
    </div>
    <div class="rec-card">
      <div class="rec-num">2</div>
      <h4>Simplify the Process</h4>
      <p><strong>From The Wall:</strong> Bhargava &amp; Manoli (2015) showed simplifying benefit language
      increased claiming by 6-8 pp. Show estimated benefit amounts, use plain language, reduce paperwork.
      35% of approved beneficiaries never exercised eligibility.</p>
    </div>
    <div class="rec-card">
      <div class="rec-num">3</div>
      <h4>Arabic-Language Services at NII</h4>
      <p><strong>From The Wall:</strong> Alhuzeel (2023) identified multi-level barriers —
      Hebrew-only forms, excessive bureaucracy, and lack of Arabic-language information about rights.
      Ensure medical committees include Arabic-speaking professionals.
      Provide all forms, notifications, and digital services in Arabic.</p>
    </div>
    <div class="rec-card">
      <div class="rec-num">4</div>
      <h4>Break the Intergenerational Cycle</h4>
      <p><strong>From Q2:</strong> settlements flagged for both child and adult disability (&rho; = {rho_overall})
      need integrated family-level interventions, not separate programs.
      The pattern persists even in affluent areas (&rho; = {rho_center} in the center).</p>
    </div>
    <div class="rec-card">
      <div class="rec-num">5</div>
      <h4>Annual Monitoring Dashboard</h4>
      <p><strong>From Q3:</strong> re-run the model annually. Settlements with persistent positive gaps
      across multiple years should be prioritized for field investigation.
      Track frontline communities ({n_frontline_panel} settlements) for post-conflict effects.</p>
    </div>
  </div>

  <div class="insight-callout" style="margin-top:24px;">
    <strong>Summary:</strong> Four questions, one conclusion: disability benefit
    under-utilization in Israel is concentrated, intergenerational, structural,
    and invisible to standard data. Our models explain ~{round(best_r2_val * 100)}%
    of the variation overall (R&sup2; = {best_r2_val}) but hit a wall for Arab
    settlements (R&sup2; = {arab_best_r2:.2f}). The barriers are not in the data &mdash;
    they are linguistic, institutional, and cultural.
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
const hrBoxplotData = {json.dumps(hr_boxplot_data)};
const arabChartData = {json.dumps(arab_chart_data)};
const temporalChartData = {json.dumps(temporal_chart_data)};
const distScatterData = {json.dumps(dist_scatter_data)};
const trendData = {json.dumps(trend_data)};
const gapScatterData = {json.dumps(gap_scatter_data)};
// ── Tooltip defaults ──
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 13;
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

// ── 2b. Ensemble Gap Scatter (Expected vs Actual, ±1.5σ residual) ──
(function() {{
  const segCfg = {{
    "Normal":                    {{ color: "rgba(76,114,176,0.5)",   border: "#4C72B0", radius: 5, style: "circle" }},
    "Potential Under-utilization":{{ color: "rgba(85,168,104,0.85)", border: "#55A868", radius: 9, style: "rectRot" }},
    "Hidden Burden":             {{ color: "rgba(196,78,82,0.85)",   border: "#C44E52", radius: 9, style: "crossRot" }}
  }};
  const order = ["Normal", "Potential Under-utilization", "Hidden Burden"];
  const datasets = order.filter(seg => gapScatterData[seg] && gapScatterData[seg].x.length > 0).map(seg => {{
    const d = gapScatterData[seg];
    const cfg = segCfg[seg];
    return {{
      label: seg + " (" + d.x.length + ")",
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
  const gapThreshold = {round(gap_threshold, 2)};

  const gapDiagonalPlugin = {{
    id: "gapDiagonal",
    afterDraw(chart) {{
      const {{ ctx, scales: {{ x: xScale, y: yScale }} }} = chart;
      ctx.save();
      // Main diagonal (perfect prediction)
      ctx.beginPath();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = "rgba(0,0,0,0.3)";
      ctx.lineWidth = 1.5;
      ctx.moveTo(xScale.getPixelForValue(diagMin), yScale.getPixelForValue(diagMin));
      ctx.lineTo(xScale.getPixelForValue(diagMax), yScale.getPixelForValue(diagMax));
      ctx.stroke();
      // Upper threshold (hidden burden zone: actual >> expected)
      ctx.beginPath();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "rgba(196,78,82,0.3)";
      ctx.lineWidth = 1;
      ctx.moveTo(xScale.getPixelForValue(diagMin), yScale.getPixelForValue(diagMin + gapThreshold));
      ctx.lineTo(xScale.getPixelForValue(diagMax), yScale.getPixelForValue(diagMax + gapThreshold));
      ctx.stroke();
      // Lower threshold (under-utilization zone: actual << expected)
      ctx.beginPath();
      ctx.strokeStyle = "rgba(85,168,104,0.3)";
      ctx.moveTo(xScale.getPixelForValue(diagMin), yScale.getPixelForValue(diagMin - gapThreshold));
      ctx.lineTo(xScale.getPixelForValue(diagMax), yScale.getPixelForValue(diagMax - gapThreshold));
      ctx.stroke();
      ctx.restore();
    }}
  }};

  window.__ensembleChart = new Chart(document.getElementById("chart_gap_scatter"), {{
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
                "Expected: " + pt.x.toFixed(2) + "%",
                "Actual: " + pt.y.toFixed(2) + "%",
                "Gap: " + (pt.gap >= 0 ? "+" : "") + pt.gap.toFixed(2) + " pp"
              ];
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: "Expected Rate (%)", font: {{ weight: "600" }} }},
          grid: {{ color: "rgba(0,0,0,0.06)" }}
        }},
        y: {{
          title: {{ display: true, text: "Actual Rate (%)", font: {{ weight: "600" }} }},
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

  window.__olsChart = new Chart(document.getElementById("chart_anomaly_scatter"), {{
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

// ── 4. Intergenerational Dual-Panel Scatter ──
(function() {{
  // SES-based color: red (1) → yellow (5) → green (10)
  function sesColor(cluster) {{
    const colors = [
      "#d73027","#f46d43","#fdae61","#fee08b","#ffffbf",
      "#d9ef8b","#a6d96a","#66bd63","#1a9850","#006837"
    ];
    const idx = Math.max(0, Math.min(9, (cluster || 5) - 1));
    return colors[idx];
  }}

  const canvasIds = ["chart_inter_periph", "chart_inter_nonperiph"];
  interScatterData.groups.forEach((g, gIdx) => {{
    // Scatter points colored by SES
    const points = g.child_rate.map((c, i) => ({{
      x: c, y: g.adult_rate[i], name: g.names[i], ses: g.ses_cluster[i]
    }}));
    const bgColors = points.map(p => sesColor(p.ses));

    const datasets = [{{
      label: g.name + " (N=" + g.n + ", r=" + g.r + ")",
      data: points,
      backgroundColor: bgColors,
      borderColor: bgColors.map(c => c),
      pointRadius: 5,
      pointHoverRadius: 8,
      order: 2
    }}];

    // Add OLS trend line
    if (g.trend) {{
      datasets.push({{
        label: "OLS (slope=" + g.trend.slope + ")",
        data: [{{ x: g.trend.x0, y: g.trend.y0 }}, {{ x: g.trend.x1, y: g.trend.y1 }}],
        type: "line",
        borderColor: "rgba(0,0,0,0.6)",
        borderWidth: 2,
        borderDash: [6, 4],
        pointRadius: 0,
        fill: false,
        order: 1
      }});
    }}

    new Chart(document.getElementById(canvasIds[gIdx]), {{
      type: "scatter",
      data: {{ datasets }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ position: "top", labels: {{ padding: 10, font: {{ size: 11 }} }} }},
          tooltip: {{
            callbacks: {{
              label: ctx => {{
                const pt = ctx.raw;
                if (!pt.name) return "";
                return [
                  pt.name,
                  "Child rate: " + pt.x.toFixed(2) + "%",
                  "Adult rate: " + pt.y.toFixed(2) + "%",
                  "SES cluster: " + (pt.ses || "?")
                ];
              }}
            }}
          }}
        }},
        scales: {{
          x: {{
            title: {{ display: true, text: "Child Disability Rate (%)", font: {{ size: 11, weight: "600" }} }},
            grid: {{ color: "rgba(0,0,0,0.06)" }}
          }},
          y: {{
            title: {{ display: true, text: "Adult Disability Rate (%)", font: {{ size: 11, weight: "600" }} }},
            grid: {{ color: "rgba(0,0,0,0.06)" }}
          }}
        }}
      }}
    }});
  }});
}})();

// ── 4b. High-Risk Cluster Boxplot (median comparison) ──
(function() {{
  function median(arr) {{
    if (!arr.length) return 0;
    const s = arr.slice().sort((a,b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 ? s[mid] : (s[mid-1] + s[mid]) / 2;
  }}
  const labels = hrBoxplotData.indicators;
  const hrMedians = hrBoxplotData.high_risk.map(median);
  const restMedians = hrBoxplotData.rest.map(median);
  const hrNorm = hrMedians.map((v, i) => restMedians[i] !== 0 ? Math.round(v / restMedians[i] * 100) : 0);
  const restNorm = restMedians.map(() => 100);

  new Chart(document.getElementById("chart_hr_boxplot"), {{
    type: "bar",
    data: {{
      labels: labels,
      datasets: [
        {{
          label: "Rest of Settlements (= 100%)",
          data: restNorm,
          backgroundColor: "rgba(76,114,176,0.7)",
          borderRadius: 4,
          barPercentage: 0.75,
          categoryPercentage: 0.65
        }},
        {{
          label: "High-Risk Cluster",
          data: hrNorm,
          backgroundColor: "rgba(244,63,94,0.75)",
          borderRadius: 4,
          barPercentage: 0.75,
          categoryPercentage: 0.65
        }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      plugins: {{
        legend: {{ position: "top", labels: {{ padding: 14 }} }},
        tooltip: {{
          callbacks: {{
            label: ctx => {{
              const idx = ctx.dataIndex;
              const isHR = ctx.datasetIndex === 1;
              const raw = isHR ? hrMedians[idx] : restMedians[idx];
              return ctx.dataset.label + ": " + raw.toFixed(1) + " (" + ctx.parsed.x + "% of baseline)";
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          title: {{ display: true, text: "% of Rest-of-Settlements Median", font: {{ weight: "600" }} }},
          grid: {{ color: "rgba(0,0,0,0.06)" }},
          suggestedMin: 0,
          suggestedMax: 200
        }},
        y: {{
          grid: {{ display: false }}
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

// ── Settlement Search ──
(function() {{
  function setupSearch(inputId, resultId, chart) {{
    const input = document.getElementById(inputId);
    const result = document.getElementById(resultId);
    if (!input || !chart) return;

    // Collect all points with names
    const allPoints = [];
    chart.data.datasets.forEach((ds, dsIdx) => {{
      ds.data.forEach((pt, ptIdx) => {{
        if (pt.name) allPoints.push({{ dsIdx, ptIdx, name: pt.name, pt }});
      }});
    }});

    // Store original radii
    const origRadii = chart.data.datasets.map(ds => ds.pointRadius);
    const origBg = chart.data.datasets.map(ds =>
      Array.isArray(ds.backgroundColor) ? [...ds.backgroundColor] : ds.backgroundColor
    );

    input.addEventListener("input", function() {{
      const q = this.value.trim().toLowerCase();
      if (!q) {{
        // Reset
        chart.data.datasets.forEach((ds, i) => {{
          ds.pointRadius = origRadii[i];
          ds.backgroundColor = origBg[i];
        }});
        result.textContent = "";
        chart.update("none");
        return;
      }}

      const matches = allPoints.filter(p => p.name.toLowerCase().includes(q));

      // Dim all points
      chart.data.datasets.forEach((ds, i) => {{
        if (typeof origRadii[i] === "number") {{
          ds.pointRadius = origRadii[i];
        }}
      }});

      // Build highlight set
      const highlightSet = new Set(matches.map(m => m.dsIdx + ":" + m.ptIdx));

      chart.data.datasets.forEach((ds, dsIdx) => {{
        const newBg = [];
        const newRadius = [];
        ds.data.forEach((pt, ptIdx) => {{
          const key = dsIdx + ":" + ptIdx;
          if (highlightSet.has(key)) {{
            newBg.push("#FF0000");
            newRadius.push(14);
          }} else {{
            const orig = Array.isArray(origBg[dsIdx]) ? origBg[dsIdx][ptIdx] : origBg[dsIdx];
            newBg.push(typeof orig === "string" ? orig.replace(/[\d.]+\)$/, "0.15)") : "rgba(150,150,150,0.15)");
            newRadius.push(typeof origRadii[dsIdx] === "number" ? origRadii[dsIdx] : 4);
          }}
        }});
        ds.backgroundColor = newBg;
        ds.pointRadius = newRadius;
      }});

      if (matches.length === 1) {{
        const m = matches[0];
        const p = m.pt;
        const info = p.gap !== undefined
          ? m.name + ": expected=" + p.x.toFixed(1) + "%, actual=" + p.y.toFixed(1) + "%, gap=" + (p.gap >= 0 ? "+" : "") + p.gap.toFixed(1)
          : p.res !== undefined
            ? m.name + ": predicted=" + p.x.toFixed(1) + "%, actual=" + p.y.toFixed(1) + "%, residual=" + (p.res >= 0 ? "+" : "") + p.res.toFixed(1)
            : m.name;
        result.textContent = info;
      }} else if (matches.length > 1) {{
        result.textContent = matches.length + " matches: " + matches.slice(0, 3).map(m => m.name).join(", ") + (matches.length > 3 ? "..." : "");
      }} else {{
        result.textContent = "No matches";
      }}

      chart.update("none");
    }});
  }}

  // Wait a tick for charts to be ready
  setTimeout(function() {{
    setupSearch("search_ensemble", "search_ensemble_result", window.__ensembleChart);
    setupSearch("search_ols", "search_ols_result", window.__olsChart);
  }}, 100);
}})();

// ── Explorer table search & filter ──
(function() {{
  const search = document.getElementById("table_search");
  const filterGroup = document.getElementById("table_filter_group");
  const filterStatus = document.getElementById("table_filter_status");
  const table = document.getElementById("explorer_table");
  if (!search || !table) return;
  const rows = Array.from(table.querySelectorAll("tbody tr"));

  const countEl = document.getElementById("table_count");
  const thPred = document.getElementById("th_pred");
  const thGap = document.getElementById("th_gap");

  function applyFilters() {{
    const q = search.value.trim().toLowerCase();
    const grp = filterGroup.value;
    const stat = filterStatus.value;
    const isOls = stat.startsWith("ols-");
    let visible = 0;

    // Switch column headers based on method
    if (thPred) thPred.innerHTML = isOls
      ? 'Expected (OLS) <span class="sort-arrow">&#9650;&#9660;</span>'
      : 'Expected (Ensemble) <span class="sort-arrow">&#9650;&#9660;</span>';
    if (thGap) thGap.innerHTML = isOls
      ? 'Gap (OLS) <span class="sort-arrow">&#9650;&#9660;</span>'
      : 'Gap (Ensemble) <span class="sort-arrow">&#9650;&#9660;</span>';

    rows.forEach(tr => {{
      const cells = tr.querySelectorAll("td");
      const name = (cells[0] || {{}}).textContent.toLowerCase();
      const group = (cells[5] || {{}}).textContent.trim();
      const ensStatus = (cells[6] || {{}}).innerHTML;
      const olsStatus = (cells[7] || {{}}).innerHTML;

      // Switch displayed values based on method
      const predCell = cells[2];
      const gapCell = cells[3];
      if (isOls) {{
        const olsPred = tr.dataset.olsPred;
        const olsGap = tr.dataset.olsGap;
        if (predCell && olsPred) predCell.textContent = olsPred + "%";
        if (gapCell && olsGap) {{
          gapCell.textContent = olsGap;
          gapCell.style.color = parseFloat(olsGap) > 0 ? "#55A868" : "#C44E52";
        }}
      }} else {{
        const ensPred = tr.dataset.ensPred;
        const ensGap = tr.dataset.ensGap;
        if (predCell && ensPred) predCell.textContent = ensPred + "%";
        if (gapCell && ensGap) {{
          gapCell.textContent = ensGap;
          gapCell.style.color = parseFloat(ensGap) > 0 ? "#55A868" : "#C44E52";
        }}
      }}

      let show = true;
      if (q && !name.includes(q)) show = false;
      if (grp && group !== grp) show = false;
      if (stat === "ens-under" && !ensStatus.includes("Under-util")) show = false;
      if (stat === "ens-hidden" && !ensStatus.includes("Hidden")) show = false;
      if (stat === "ols-under" && !olsStatus.includes("Under-util")) show = false;
      if (stat === "ols-hidden" && !olsStatus.includes("Hidden")) show = false;
      if (stat === "any-flag" && !ensStatus.includes("Under-util") && !ensStatus.includes("Hidden") && !olsStatus.includes("Under-util") && !olsStatus.includes("Hidden")) show = false;
      tr.style.display = show ? "" : "none";
      if (show) visible++;
    }});
    if (countEl) countEl.textContent = visible + " of " + rows.length + " settlements";
  }}
  applyFilters();

  search.addEventListener("input", applyFilters);
  filterGroup.addEventListener("change", applyFilters);
  filterStatus.addEventListener("change", applyFilters);
}})();
</script>

</body>
</html>
"""

# ── Write output ──────────────────────────────────────────────────────────────

out_path = PROJECT_ROOT / "presentation_phase2.html"
out_path.write_text(html_output, encoding="utf-8")
print(f"\nDone! Written to {out_path}")
print(f"  File size: {out_path.stat().st_size / 1024:.0f} KB")
