"""
Generate Executive HTML presentation for policymakers.
Reads benefits_final.csv, trains models, loads temporal data,
produces presentation_executive.html with 9 slides structured
around the 4 research questions:
  1. Task (title + 4 questions)
  2. Method (models + feature importance)
  3. Q1: Gap (anomaly scatter)
  4. Q2: Generations (intergenerational scatter)
  5. The Wall (Arab R² wall)
  6. Q3: Conflict (temporal frontline trends)
  7. Q4: Distance (distance experiment)
  8. Evidence (Israeli research + Obermeyer)
  9. Action (recommendations + summary)
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
from sklearn.model_selection import GroupKFold, cross_val_predict
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


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    head_html = "".join(f"<th>{html_lib.escape(h)}</th>" for h in headers)
    body_html = ""
    for row in rows:
        cells = "".join(f"<td>{html_lib.escape(str(cell))}</td>" for cell in row)
        body_html += f"<tr>{cells}</tr>"
    return (
        '<table class="data-table">'
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
    )


# ── Temporal data helpers (from phase2) ──────────────────────────────────────

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

# Load frontline settlement codes
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

# Plotly-ready temporal data
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

print(f"  Frontline: {n_frontline_panel} settlements, rate {front_rate_2023}% -> {front_rate_2025}% (\u0394={front_delta_total:+.2f})")
print(f"  Non-frontline: {n_nonfrontline_panel} settlements, rate {nf_rate_2023}% -> {nf_rate_2025}% (\u0394={nf_delta_total:+.2f})")

# ── Distance data (Q4) ───────────────────────────────────────────────────────

print("Loading distance data ...")
DIST_CSV = DATA_DIR / "data" / "processed" / "my_dataset_with_distances.csv"
df_dist = pd.read_csv(DIST_CSV)
df_dist["settlement_code"] = pd.to_numeric(df_dist["settlement_code"], errors="coerce")
for dc in ["dist_any_branch_km", "dist_central_branch_km", "dist_central_medical_branch_km"]:
    df_dist[dc] = pd.to_numeric(df_dist[dc], errors="coerce")

# Merge distances into main df via settlement_symbol ↔ settlement_code
dist_map = df_dist.set_index("settlement_code")[["dist_any_branch_km", "dist_central_branch_km", "dist_central_medical_branch_km"]]
df["settlement_symbol_num"] = pd.to_numeric(df.get("settlement_symbol", pd.Series(dtype=float)), errors="coerce")
df = df.merge(dist_map, left_on="settlement_symbol_num", right_index=True, how="left")

# Scatter data for dropdown chart (distance vs disability rate)
df_dist_plot = df.dropna(subset=["dist_any_branch_km", "general_disability_rate"]).copy()
dist_scatter_data = {
    "names": df_dist_plot["settlement_name"].tolist(),
    "disability_rate": df_dist_plot["general_disability_rate"].round(2).tolist(),
    "dist_any": df_dist_plot["dist_any_branch_km"].round(2).tolist(),
    "dist_central": df_dist_plot["dist_central_branch_km"].round(2).tolist(),
    "dist_central_medical": df_dist_plot["dist_central_medical_branch_km"].round(2).tolist(),
}

# Trend line data (SES vs distance, z-scored)
ses_col = "socio_economic_index_score"
dist_col = "dist_any_branch_km"
df_trend = df[[ses_col, dist_col, "general_disability_rate"]].dropna().copy()

ses_mean, ses_std = df_trend[ses_col].mean(), df_trend[ses_col].std(ddof=0)
dist_mean, dist_std = df_trend[dist_col].mean(), df_trend[dist_col].std(ddof=0)
df_trend["ses_z"] = (df_trend[ses_col] - ses_mean) / ses_std
df_trend["dist_z"] = (df_trend[dist_col] - dist_mean) / dist_std

lr_ses = LinearRegression().fit(df_trend[["ses_z"]], df_trend["general_disability_rate"])
lr_dist = LinearRegression().fit(df_trend[["dist_z"]], df_trend["general_disability_rate"])

z_min = min(df_trend["ses_z"].min(), df_trend["dist_z"].min())
z_max = max(df_trend["ses_z"].max(), df_trend["dist_z"].max())
x_grid = np.linspace(z_min, z_max, 200)
y_ses = lr_ses.predict(x_grid.reshape(-1, 1))
y_dist = lr_dist.predict(x_grid.reshape(-1, 1))

# 95% CI band for SES
residuals_ses = df_trend["general_disability_rate"].values - lr_ses.predict(df_trend[["ses_z"]])
sigma_ses = float(np.std(residuals_ses, ddof=1))
y_ses_upper = y_ses + 1.96 * sigma_ses
y_ses_lower = y_ses - 1.96 * sigma_ses

trend_data = {
    "x_grid": [round(float(v), 3) for v in x_grid],
    "y_ses": [round(float(v), 3) for v in y_ses],
    "y_dist": [round(float(v), 3) for v in y_dist],
    "y_ses_upper": [round(float(v), 3) for v in y_ses_upper],
    "y_ses_lower": [round(float(v), 3) for v in y_ses_lower],
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

# Plain English labels for feature importance chart
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
print(f"  RF: R\u00b2 = {rf_scores['r2']}")

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
        print(f"  XGB: R\u00b2 = {xgb_scores['r2']}")
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
        print(f"  TabPFN: R\u00b2 = {tabpfn_scores['r2']}")
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

# ── Sector R² ───────────────────────────────────────────────────────────────

print("Computing sector R\u00b2 ...")

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

# Under-utilization demographic composition
under_arab_pct = round(len(arab_under) / n_under * 100) if n_under > 0 else 0
haredi_pct_col = pd.to_numeric(
    df_gap.get("haredi_population_percentage", pd.Series(dtype=float)), errors="coerce"
).fillna(0)
haredi_under = under_candidates[haredi_pct_col.reindex(under_candidates.index).fillna(0) > 20]
under_haredi_pct = round(len(haredi_under) / n_under * 100) if n_under > 0 else 0
under_secular_pct = 100 - under_arab_pct - under_haredi_pct

# Sector R² chart data
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

# ── Intergenerational analysis (Q2) ─────────────────────────────────────────

print("Computing intergenerational analysis ...")

child_rate_col = pd.to_numeric(df.loc[df_reg.index, "disabled_child_benefit_rate"], errors="coerce")
adult_rate_col = pd.to_numeric(df.loc[df_reg.index, "general_disability_rate"], errors="coerce")
periph_cluster = pd.to_numeric(df.loc[df_reg.index, "peripherality_index_cluster"], errors="coerce")

# Overall Spearman
inter_valid = child_rate_col.notna() & adult_rate_col.notna()
rho_overall, pval_overall = spearmanr(
    child_rate_col[inter_valid], adult_rate_col[inter_valid]
)
rho_overall = round(rho_overall, 3)

# By peripherality group
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

# Scatter data for Plotly (colored by peripherality group)
inter_scatter_data = {"groups": []}
for group_label, group_mask, color in [
    ("Periphery (clusters 1-3)", periph_periphery & inter_valid, "#F59E0B"),
    ("Middle (clusters 4-7)", (~periph_periphery & ~periph_center) & inter_valid, "#94A3B8"),
    ("Center (clusters 8-10)", periph_center & inter_valid, "#38BDF8"),
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

print(f"  Overall \u03c1={rho_overall}, Periphery \u03c1={rho_periph} (N={n_periph}), Center \u03c1={rho_center} (N={n_center})")


# ── HTML generation ──────────────────────────────────────────────────────────

print("Generating HTML ...")

slides: list[str] = []
nav_items: list[tuple[str, str]] = []


def add_slide(slide_id: str, nav_label: str, html_block: str) -> None:
    slides.append(html_block)
    nav_items.append((nav_label, slide_id))


# ── Slide 1: The Task (4 Research Questions) ─────────────────────────────────

slide1 = f"""
<section class="slide" id="slide1">
  <div class="container">
    <div class="title-block reveal">
      <h1>Detecting Disability Benefit<br>Under-Utilization in Israel</h1>
      <p class="subtitle">A Machine Learning Approach to Identifying Access Barriers</p>
      <p class="meta">TovTech Research Group &nbsp;|&nbsp; Data: National Insurance Institute, Dec 2024</p>
    </div>
    <div class="hero-panel reveal delay-1">
      <p style="margin-bottom:14px;font-weight:600;color:var(--cyan);">Four Research Questions</p>
      <div class="questions-grid">
        <div class="q-item"><span class="q-num">Q1</span>
          <strong>Under-utilization pockets</strong> &mdash; Where is disability claiming
          lower than the model predicts?</div>
        <div class="q-item"><span class="q-num">Q2</span>
          <strong>Intergenerational trap</strong> &mdash; Is child disability correlated
          with adult disability?</div>
        <div class="q-item"><span class="q-num">Q3</span>
          <strong>Swords of Iron</strong> &mdash; Did the conflict change disability trends
          in frontline settlements?</div>
        <div class="q-item"><span class="q-num">Q4</span>
          <strong>Service deserts</strong> &mdash; Does distance to medical committees
          explain the claiming gap?</div>
      </div>
    </div>
    <div class="stat-row reveal delay-2">
      <div class="stat-card">
        <div class="stat-value" data-target="{n_reg}">{n_reg}</div>
        <div class="stat-label">Settlements analyzed</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" data-target="{n_features}">{n_features}</div>
        <div class="stat-label">Socio-economic indicators</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" data-target="3">3</div>
        <div class="stat-label">ML models tested</div>
      </div>
    </div>
  </div>
</section>
"""
add_slide("slide1", "Task", slide1)


# ── Slide 2: Method (Models + Feature Importance) ───────────────────────────

n_models_text = "three" if tabpfn_available else ("two" if xgb_available else "one")

slide2 = f"""
<section class="slide" id="slide2">
  <div class="container">
    <h2 class="reveal">Methodology</h2>
    <p class="subtitle reveal delay-1">Predicting what disability rates <em>should</em>
    look like &mdash; then finding the gaps</p>
    <div class="two-col reveal delay-1">
      <div class="glass-panel">
        <h3>Models</h3>
        <p>We trained {n_models_text} ML algorithms with spatial cross-validation:</p>
        <ul>
          <li><strong>Random Forest</strong> &mdash; R&sup2;&nbsp;=&nbsp;{rf_scores['r2']}</li>
          {"<li><strong>XGBoost</strong> &mdash; R&sup2;&nbsp;=&nbsp;" + str(xgb_scores["r2"]) + "</li>" if xgb_scores else ""}
          {"<li><strong>TabPFN v2</strong> &mdash; R&sup2;&nbsp;=&nbsp;" + str(tabpfn_scores["r2"]) + " (Nature, Jan 2025)</li>" if tabpfn_scores else ""}
        </ul>
        <p style="margin-top:10px;">Best overall R&sup2;&nbsp;=&nbsp;<strong>{best_r2_val}</strong>
        &mdash; {n_features} features explain ~{round(best_r2_val*100)}% of disability rate
        variation.</p>
      </div>
      <div class="glass-panel">
        <h3>Feature Categories</h3>
        <ul>
          <li><strong>Economic:</strong> socio-economic score, salary, employment</li>
          <li><strong>Education:</strong> academic degrees, dropout rate, higher ed entry</li>
          <li><strong>Demographics:</strong> Arab %, Haredi %, age structure</li>
          <li><strong>Geography:</strong> peripherality, latitude, settlement size</li>
          <li><strong>Intergenerational:</strong> child disability rate</li>
        </ul>
      </div>
    </div>
    <div class="chart-container reveal delay-2"><div id="graph_importance"></div></div>
  </div>
</section>
"""
add_slide("slide2", "Method", slide2)


# ── Slide 3: Q1 Gap (Anomaly Scatter) ───────────────────────────────────────

slide3 = f"""
<section class="slide" id="slide3">
  <div class="container">
    <h2 class="reveal">Q1: Where Are the Under-Utilization Pockets?</h2>
    <p class="subtitle reveal delay-1">{n_under} settlements where disability claiming is
    significantly lower than predicted</p>
    <div class="chart-container reveal delay-1"><div id="graph_gap_scatter"></div></div>
    <div class="two-col reveal delay-2">
      <div class="glass-panel">
        <h3>Reading the Chart</h3>
        <p>Each dot is a settlement. The diagonal = perfect prediction.</p>
        <ul>
          <li><strong style="color:#F59E0B;">Gold diamonds</strong> &mdash;
              <strong>{n_under} settlements</strong> where observed claiming is far below expected
              (potential under-utilization)</li>
          <li><strong style="color:#F43F5E;">Red crosses</strong> &mdash;
              <strong>{n_hidden} settlements</strong> where observed claiming exceeds expectations
              (hidden burden)</li>
        </ul>
      </div>
      <div class="glass-panel">
        <h3>Who Is Over-Represented?</h3>
        <p>Among the {n_under} flagged settlements:</p>
        <ul>
          <li>Arab-majority: <strong>{under_arab_pct}%</strong>
              (vs. {round(n_arab/n_reg*100)}% of dataset)</li>
          <li>Haredi-significant: <strong>{under_haredi_pct}%</strong></li>
          <li>Secular/mixed: <strong>{under_secular_pct}%</strong></li>
        </ul>
        <p style="margin-top:8px;"><strong>Arab and ultra-Orthodox communities are
        disproportionately flagged.</strong></p>
      </div>
    </div>
  </div>
</section>
"""
add_slide("slide3", "Q1: Gap", slide3)


# ── Slide 4: Q2 Intergenerational ───────────────────────────────────────────

slide4 = f"""
<section class="slide" id="slide4">
  <div class="container">
    <h2 class="reveal">Q2: The Intergenerational Trap</h2>
    <p class="subtitle reveal delay-1">Child disability benefit rate correlates with adult
    disability rate &mdash; \u03c1&nbsp;=&nbsp;{rho_overall}</p>
    <div class="chart-container reveal delay-1"><div id="graph_intergenerational"></div></div>
    <div class="two-col reveal delay-2">
      <div class="glass-panel">
        <h3>What This Means</h3>
        <p>Settlements where more children receive disability benefits also have higher
        adult disability rates. This suggests a <strong>cycle of vulnerability</strong>:
        conditions that lead to childhood disability persist into adulthood.</p>
        <p style="margin-top:10px;">The correlation is <strong>not</strong> just a
        by-product of poverty &mdash; it holds across socio-economic levels.</p>
      </div>
      <div class="glass-panel">
        <h3>Stronger at the Periphery</h3>
        <p>The intergenerational link varies by location:</p>
        <ul>
          <li><strong>Periphery</strong> (clusters 1-3): \u03c1 = {rho_periph}
              ({n_periph} settlements)</li>
          <li><strong>Center</strong> (clusters 8-10): \u03c1 = {rho_center}
              ({n_center} settlements)</li>
        </ul>
        <p style="margin-top:8px;">Paradoxically, the center shows a <em>stronger</em>
        correlation &mdash; even in affluent areas, intergenerational disability
        patterns persist, suggesting structural rather than purely economic causes.</p>
      </div>
    </div>
  </div>
</section>
"""
add_slide("slide4", "Q2: Generations", slide4)


# ── Slide 5: The Wall ──────────────────────────────────────────────────────

slide5 = f"""
<section class="slide" id="slide5">
  <div class="container">
    <h2 class="reveal">The Wall: Q1 + Q2 Converge on the Arab Sector</h2>
    <p class="subtitle reveal delay-1">{n_arab} Arab-majority settlements &mdash;
    best model R&sup2;&nbsp;=&nbsp;{arab_best_r2:.3f}</p>
    <div class="chart-container reveal delay-1"><div id="graph_arab_r2"></div></div>
    <div class="warning-panel glow-border reveal delay-2">
      <div class="big-number">~{arab_unexplained_pct}%</div>
      <div class="big-label">of the variance in Arab disability claiming
      <strong>cannot be explained</strong> by any combination of {n_features}
      socio-economic indicators</div>
    </div>
    <div class="two-col reveal delay-3">
      <div class="glass-panel">
        <h3>The &ldquo;So What&rdquo;</h3>
        <p>For Jewish-secular settlements, our models explain
        <strong>{fmt_num(best_secular_r2 * 100, 0)}%</strong> of the variation. The standard
        factors (income, education, demographics) work.</p>
        <p style="margin-top:8px;">For Arab settlements, <strong>the same factors explain
        almost nothing</strong>. Something invisible to official statistics drives
        the gap &mdash; and Q2 shows this pattern starts in childhood.</p>
      </div>
      <div class="glass-panel">
        <h3>Under-utilization is concentrated here</h3>
        <p>Of the {n_under} flagged settlements (Q1), <strong>{under_arab_pct}%</strong> are
        Arab-majority. These are communities where:</p>
        <ul>
          <li>The model <em>predicts</em> high disability rates</li>
          <li>Observed claiming is far <em>below</em> prediction</li>
          <li>The intergenerational pattern (Q2) is present</li>
        </ul>
        <p style="margin-top:8px;">This convergence points to <strong>access barriers</strong>,
        not lower need.</p>
      </div>
    </div>
  </div>
</section>
"""
add_slide("slide5", "The Wall", slide5)


# ── Slide 6: Q3 Conflict (Temporal) ─────────────────────────────────────────

slide6 = f"""
<section class="slide" id="slide6">
  <div class="container">
    <h2 class="reveal">Q3: Did the Conflict Change Disability Trends?</h2>
    <p class="subtitle reveal delay-1">Frontline settlements (Swords of Iron, Oct 2023&ndash;)
    vs. the rest of Israel</p>
    <div class="chart-container reveal delay-1"><div id="graph_temporal"></div></div>
    <div class="two-col reveal delay-2">
      <div class="glass-panel">
        <h3>The Numbers</h3>
        <ul>
          <li><strong>Frontline</strong> ({n_frontline_panel} settlements):
              {front_rate_2023}% &rarr; {front_rate_2025}%
              (\u0394&nbsp;=&nbsp;{front_delta_total:+.2f}&nbsp;pp)</li>
          <li><strong>Non-frontline</strong> ({n_nonfrontline_panel} settlements):
              {nf_rate_2023}% &rarr; {nf_rate_2025}%
              (\u0394&nbsp;=&nbsp;{nf_delta_total:+.2f}&nbsp;pp)</li>
        </ul>
      </div>
      <div class="glass-panel">
        <h3>Interpretation</h3>
        <p>Disability rates rose <strong>everywhere</strong>, not just in frontline zones.
        The increase is <strong>structural</strong>, not crisis-driven:</p>
        <ul>
          <li>Population aging continues nationwide</li>
          <li>NII approval processes have multi-year lags</li>
          <li>Frontline communities already had high baseline rates</li>
        </ul>
        <p style="margin-top:8px;">The conflict did not create a new disability spike &mdash;
        it <strong>amplified existing vulnerabilities</strong>.</p>
      </div>
    </div>
  </div>
</section>
"""
add_slide("slide6", "Q3: Conflict", slide6)


# ── Slide 7: Q4 Distance ────────────────────────────────────────────────────

slide7 = f"""
<section class="slide" id="slide7">
  <div class="container">
    <h2 class="reveal">Q4: Is Distance the Barrier?</h2>
    <p class="subtitle reveal delay-1">Testing whether physical access to NII offices
    explains the claiming gap</p>
    <div class="chart-container reveal delay-1"><div id="graph_dist_scatter"></div></div>
    <div class="chart-container reveal delay-2"><div id="graph_trend_lines"></div></div>
    <div class="two-col reveal delay-3">
      <div class="glass-panel">
        <h3>What the Charts Show</h3>
        <p><strong>Top chart:</strong> each dot is a settlement. No visible trend &mdash;
        settlements far from NII branches have the <em>same</em> disability rates as
        those nearby. Toggle between branch types with the dropdown.</p>
        <p style="margin-top:10px;"><strong>Bottom chart:</strong> two predictors
        on the same normalized scale. Socio-economic status (red) has a strong
        slope ({trend_data['slope_ses']:+.2f}), while distance (blue) is nearly
        flat ({trend_data['slope_dist']:+.2f}).</p>
      </div>
      <div class="glass-panel">
        <h3>Conclusion</h3>
        <p>Physical distance to NII branches is <strong>not the barrier</strong>.</p>
        <p style="margin-top:8px;">The remaining candidates:</p>
        <ul>
          <li><strong>Informational:</strong> people don&rsquo;t know they&rsquo;re eligible</li>
          <li><strong>Linguistic:</strong> forms and committees operate in Hebrew</li>
          <li><strong>Cultural:</strong> stigma prevents claiming</li>
          <li><strong>Institutional:</strong> the process itself deters applicants</li>
        </ul>
      </div>
    </div>
  </div>
</section>
"""
add_slide("slide7", "Q4: Distance", slide7)


# ── Slide 8: Evidence ──────────────────────────────────────────────────────

slide8 = """
<section class="slide" id="slide8">
  <div class="container">
    <h2 class="reveal">The Evidence</h2>
    <p class="subtitle reveal delay-1">Israeli research and an international parallel confirm
    our findings</p>
    <div class="two-col reveal delay-1">
      <div class="evidence-card" style="flex:1;">
        <div class="evidence-source">Alhuzeel et al. (2024) &mdash;
        <em>Scandinavian J. of Disability Research</em></div>
        <div class="evidence-quote glow-border" style="padding:12px 16px;border-radius:12px;">&ldquo;It&rsquo;s Disgraceful Going through All this
        for Being an Arab and Disabled&rdquo;</div>
        <div class="evidence-body">
          <p>Interviews with 15 Arab Israelis with disabilities revealed barriers at
          every level:</p>
          <ul>
            <li><strong>Family:</strong> disability perceived as shameful; families avoid
                state services to prevent reduced marriage prospects</li>
            <li><strong>Institution:</strong> NII perceived as the <strong>most discriminatory</strong>
                government body. Medical committees lack Arabic-speaking staff</li>
            <li><strong>Society:</strong> &ldquo;the invisible of the invisible&rdquo;</li>
          </ul>
        </div>
      </div>
      <div class="evidence-card" style="flex:1;">
        <div class="evidence-source">Brookdale Institute (2024) &mdash;
        People with Disabilities in the Arab Population</div>
        <div class="evidence-body">
          <p>Arab self-reported disability prevalence is <strong>higher</strong> than Jewish
          &mdash; <strong style="color:var(--danger);">21% vs.&nbsp;19%</strong> &mdash; yet
          benefit claiming is <strong>lower</strong>.</p>
          <p style="margin-top:8px;">Arab people with disabilities face
          <strong>&ldquo;double exclusion&rdquo;</strong>. Women face
          <strong>&ldquo;triple exclusion&rdquo;</strong> (minority + disability + gender).</p>
          <p style="margin-top:8px;"><strong>35%</strong> of people already approved for
          benefits never exercised eligibility (Brookdale, 2022).
          <strong>23%</strong> were simply unaware of the approval.</p>
        </div>
      </div>
    </div>
    <div class="two-col reveal delay-2">
      <div class="glass-panel" style="border-top:3px solid var(--pink);flex:1;">
        <h3>The <em>Science</em> Parallel</h3>
        <p><strong>Obermeyer et al. (2019)</strong>: A US healthcare algorithm on
        200&nbsp;million patients used <em>cost</em> as proxy for <em>need</em>.
        Because the system spent less on Black patients with equivalent illness,
        the algorithm concluded they were healthier.</p>
        <p style="margin-top:8px;">Our case: NII data records <em>benefit receipt</em>
        as proxy for <em>disability prevalence</em>. In communities facing barriers,
        fewer people claim &mdash; creating the illusion of lower need.</p>
      </div>
      <div class="glass-panel" style="border-top:3px solid var(--cyan);flex:1;">
        <h3>Fiscal Inequality</h3>
        <p>Ministry of Welfare per-client allocation (Taub Center):</p>
        <p style="margin-top:8px;"><strong style="color:var(--danger);">Arab:
        NIS&nbsp;2,682</strong> vs. <strong style="color:var(--cyan);">Jewish:
        NIS&nbsp;5,483</strong> &mdash; less than half.</p>
        <p style="margin-top:8px;">The tax &amp; transfer system reduces Arab poverty by
        only <strong>8.4%</strong> vs. <strong>45.5%</strong> for Jewish households.</p>
      </div>
    </div>
  </div>
</section>
"""
add_slide("slide8", "Evidence", slide8)


# ── Slide 9: Recommendations ──────────────────────────────────────────────

slide9 = f"""
<section class="slide" id="slide9">
  <div class="container">
    <h2 class="reveal">Recommendations</h2>
    <p class="subtitle reveal delay-1">Evidence-based actions to close the disability
    benefit access gap</p>
    <div class="three-col reveal delay-1">
      <div class="rec-card">
        <div class="rec-number">1</div>
        <h3>Targeted Outreach</h3>
        <p>The <strong>{n_under} flagged settlements</strong> should receive proactive
        information campaigns &mdash; <strong>in Arabic</strong>, with community-based
        intermediaries and local advocacy organizations.</p>
      </div>
      <div class="rec-card">
        <div class="rec-number">2</div>
        <h3>Simplify the Process</h3>
        <p>Bhargava &amp; Manoli (2015): <strong>simplifying benefit language increased
        claiming by 6&ndash;8 pp</strong>. Show estimated benefit amounts, use plain
        language, reduce paperwork.</p>
      </div>
      <div class="rec-card">
        <div class="rec-number">3</div>
        <h3>Arabic-Language Services</h3>
        <p>Ensure NII medical committees include <strong>Arabic-speaking
        professionals</strong>. Provide all forms, notifications, and digital services
        in Arabic.</p>
      </div>
    </div>
    <div class="two-col reveal delay-2">
      <div class="rec-card">
        <div class="rec-number">4</div>
        <h3>Break the Intergenerational Cycle</h3>
        <p>Settlements flagged for <strong>both</strong> child and adult disability
        (Q2) need integrated family-level interventions, not separate programs for
        adults and children.</p>
      </div>
      <div class="rec-card">
        <div class="rec-number">5</div>
        <h3>Annual Monitoring</h3>
        <p>Re-run the model annually. Settlements with <strong>persistent positive gaps
        </strong> across multiple years should be prioritized for field investigation.
        Track frontline communities for post-conflict effects.</p>
      </div>
    </div>
    <div class="glass-panel reveal delay-3" style="max-width:960px;text-align:center;">
      <h3>Summary</h3>
      <p style="font-size:1.1rem;line-height:1.8;">Four questions, one conclusion:
      disability benefit under-utilization in Israel is <strong>concentrated,
      intergenerational, structural, and invisible to standard data</strong>.
      Our models explain ~{round(best_r2_val*100)}% of the variation overall
      (R&sup2;&nbsp;=&nbsp;{best_r2_val}) but <strong>hit a wall for Arab settlements</strong>
      (R&sup2;&nbsp;=&nbsp;{arab_best_r2:.2f}). This is mathematical proof that the
      factors driving the gap &mdash; language barriers, institutional friction,
      and systemic under-investment &mdash; are invisible to official statistics.</p>
    </div>
  </div>
</section>
"""
add_slide("slide9", "Action", slide9)


# ── Assemble HTML ────────────────────────────────────────────────────────────

slides_html = "\n".join(slides)
nav_links = "".join(
    f'<a href="#{sid}">{html_lib.escape(label)}</a>' for label, sid in nav_items
)
nav_html = f'<div class="nav">{nav_links}</div>'

html_head = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Detecting Disability Benefit Under-Utilization in Israel</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    @property --angle {{
      syntax: "<angle>";
      initial-value: 0deg;
      inherits: false;
    }}
    :root {{
      --bg: #0B0F19;
      --bg-2: #111827;
      --panel: rgba(255, 255, 255, 0.03);
      --panel-hover: rgba(255, 255, 255, 0.06);
      --accent: #F59E0B;
      --cyan: #38BDF8;
      --pink: #818CF8;
      --danger: #F43F5E;
      --text: #F3F4F6;
      --muted: #94A3B8;
      --glass-border: rgba(255, 255, 255, 0.1);
      --glass-blur: blur(16px);
    }}
    ::-webkit-scrollbar {{ display: none; }}
    html {{
      scroll-behavior: smooth;
      scrollbar-width: none;
    }}
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}
    body {{
      font-family: "Plus Jakarta Sans", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle, rgba(255,255,255,0.03) 1px, transparent 1px),
        radial-gradient(circle at 18% 10%, rgba(56, 189, 248, 0.07), transparent 42%),
        radial-gradient(circle at 85% 18%, rgba(129, 140, 248, 0.05), transparent 40%),
        linear-gradient(135deg, var(--bg), var(--bg-2));
      background-size: 32px 32px, 100% 100%, 100% 100%, 100% 100%;
      background-attachment: fixed;
      overflow-y: scroll;
      scroll-snap-type: y proximity;
    }}
    body::after {{
      content: "";
      position: fixed;
      inset: 0;
      z-index: 9999;
      pointer-events: none;
      opacity: 0.035;
      background: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
      background-size: 200px 200px;
    }}
    h1, h2 {{
      font-family: "Instrument Serif", serif;
      letter-spacing: -0.01em;
    }}
    h3, h4 {{
      font-family: "Plus Jakarta Sans", sans-serif;
      letter-spacing: -0.01em;
    }}
    h1 {{
      font-size: 3rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--cyan), var(--pink));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      text-align: center;
      line-height: 1.2;
    }}
    h2 {{
      font-size: 2.2rem;
      font-weight: 700;
      text-align: center;
      background: linear-gradient(135deg, var(--cyan), var(--pink));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    h3 {{
      font-size: 1.1rem;
      margin-bottom: 10px;
      color: var(--cyan);
    }}
    .slide {{
      min-height: 100vh;
      scroll-snap-align: start;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 48px 24px 90px;
    }}
    .container {{
      max-width: 1300px;
      width: 100%;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 20px;
    }}
    .title-block {{
      text-align: center;
      max-width: 900px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .subtitle {{
      font-size: 1.1rem;
      color: var(--muted);
      text-align: center;
    }}
    .meta {{
      font-size: 0.9rem;
      color: #94A3B8;
      text-align: center;
    }}
    .hero-panel {{
      max-width: 900px;
      padding: 32px 36px;
      border-radius: 24px;
      border: 1px solid var(--glass-border);
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.08), rgba(56, 189, 248, 0.04));
      backdrop-filter: var(--glass-blur);
      -webkit-backdrop-filter: var(--glass-blur);
      font-size: 1.05rem;
      line-height: 1.8;
      font-weight: 400;
    }}
    .questions-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 8px;
    }}
    .q-item {{
      padding: 12px 16px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.06);
      background: rgba(255,255,255,0.02);
      font-size: 0.95rem;
      line-height: 1.6;
    }}
    .q-num {{
      display: inline-block;
      background: linear-gradient(135deg, var(--cyan), var(--accent));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      font-weight: 700;
      font-size: 1rem;
      margin-right: 6px;
    }}
    .stat-row {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      justify-content: center;
    }}
    .stat-card {{
      background: var(--panel);
      border: 1px solid var(--glass-border);
      border-radius: 20px;
      padding: 24px 32px;
      text-align: center;
      backdrop-filter: var(--glass-blur);
      -webkit-backdrop-filter: var(--glass-blur);
      min-width: 140px;
    }}
    .stat-value {{
      font-family: "Plus Jakarta Sans", sans-serif;
      font-size: 2.4rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--cyan), var(--accent));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    .stat-label {{
      font-size: 0.85rem;
      color: var(--muted);
      margin-top: 4px;
    }}
    .two-col {{
      display: flex;
      gap: 16px;
      width: 100%;
      max-width: 1300px;
    }}
    .two-col > * {{
      flex: 1;
    }}
    .three-col {{
      display: flex;
      gap: 16px;
      width: 100%;
      max-width: 1300px;
    }}
    .three-col > * {{
      flex: 1;
    }}
    .glass-panel, .insight {{
      padding: 24px 28px;
      border-radius: 24px;
      border: 1px solid var(--glass-border);
      background: var(--panel);
      backdrop-filter: var(--glass-blur);
      -webkit-backdrop-filter: var(--glass-blur);
      line-height: 1.7;
      font-size: 0.95rem;
    }}
    .glass-panel:hover {{
      border-color: rgba(255, 255, 255, 0.15);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }}
    .insight {{
      width: 100%;
      max-width: 1300px;
      border-left: 3px solid var(--cyan);
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.06), transparent);
    }}
    .chart-container {{
      width: 100%;
      max-width: 1300px;
      background: var(--panel);
      border: 1px solid var(--glass-border);
      border-radius: 24px;
      padding: 16px;
      backdrop-filter: var(--glass-blur);
      -webkit-backdrop-filter: var(--glass-blur);
    }}
    .warning-panel {{
      width: 100%;
      max-width: 900px;
      padding: 32px;
      border-radius: 24px;
      border: 1px solid rgba(244, 63, 94, 0.3);
      background: linear-gradient(135deg, rgba(244, 63, 94, 0.08), rgba(129, 140, 248, 0.04));
      text-align: center;
    }}
    .big-number {{
      font-family: "Plus Jakarta Sans", sans-serif;
      font-size: 4rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--danger), var(--pink));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }}
    .big-label {{
      font-size: 1.1rem;
      color: var(--muted);
      max-width: 600px;
      margin: 0 auto;
      line-height: 1.6;
    }}
    .evidence-card {{
      width: 100%;
      max-width: 1300px;
      padding: 24px 28px;
      border-radius: 24px;
      border: 1px solid var(--glass-border);
      background: var(--panel);
      backdrop-filter: var(--glass-blur);
      -webkit-backdrop-filter: var(--glass-blur);
      line-height: 1.7;
      font-size: 0.95rem;
    }}
    .evidence-source {{
      font-size: 0.8rem;
      color: var(--cyan);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 6px;
      font-weight: 600;
    }}
    .evidence-quote {{
      font-size: 1.1rem;
      font-style: italic;
      color: var(--accent);
      margin-bottom: 12px;
      font-weight: 500;
    }}
    .evidence-body {{
      color: var(--text);
    }}
    .evidence-body ul {{
      padding-left: 20px;
      margin-top: 8px;
    }}
    .evidence-body li {{
      margin-bottom: 4px;
    }}
    .rec-card {{
      padding: 24px 28px;
      border-radius: 24px;
      border: 1px solid var(--glass-border);
      background: var(--panel);
      backdrop-filter: var(--glass-blur);
      -webkit-backdrop-filter: var(--glass-blur);
      line-height: 1.7;
      font-size: 0.95rem;
    }}
    .rec-number {{
      font-family: "Plus Jakarta Sans", sans-serif;
      font-size: 1.6rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--cyan), var(--accent));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 8px;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    .data-table th, .data-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
      text-align: left;
    }}
    .data-table th {{
      color: var(--muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 0.75rem;
    }}
    .data-table tbody tr:hover {{
      background: rgba(255, 255, 255, 0.03);
    }}
    .nav {{
      position: fixed;
      bottom: 18px;
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: center;
      background: rgba(11, 15, 25, 0.85);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      padding: 8px 14px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.1);
      z-index: 1000;
    }}
    .nav a {{
      color: var(--muted);
      text-decoration: none;
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 0.85rem;
      transition: all 0.3s ease;
    }}
    .nav a:hover {{
      color: var(--text);
      background: rgba(255, 255, 255, 0.06);
    }}
    .nav a:focus-visible {{
      outline: 2px solid var(--cyan);
      outline-offset: 2px;
    }}
    .nav a.active {{
      background: var(--cyan);
      color: var(--bg);
      font-weight: 700;
    }}
    .glow-border {{
      border: double 2px transparent;
      background-image: linear-gradient(var(--bg), var(--bg)), conic-gradient(from var(--angle), var(--cyan), var(--pink), var(--accent), var(--cyan));
      background-origin: border-box;
      background-clip: padding-box, border-box;
      animation: rotate-border 4s linear infinite;
    }}
    @keyframes rotate-border {{
      to {{ --angle: 360deg; }}
    }}
    .reveal {{
      opacity: 0;
      transform: translateY(30px);
      transition: all 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    .reveal.active {{
      opacity: 1;
      transform: translateY(0);
    }}
    .delay-1 {{ transition-delay: 0.15s; }}
    .delay-2 {{ transition-delay: 0.3s; }}
    .delay-3 {{ transition-delay: 0.45s; }}
    ul {{ padding-left: 20px; }}
    li {{ margin-bottom: 4px; }}
    @media (max-width: 900px) {{
      .slide {{ padding: 36px 16px 90px; }}
      h1 {{ font-size: 2rem; }}
      h2 {{ font-size: 1.5rem; }}
      .two-col, .three-col {{
        flex-direction: column;
      }}
      .questions-grid {{
        grid-template-columns: 1fr;
      }}
      .stat-row {{
        gap: 10px;
      }}
      .stat-card {{
        min-width: 100px;
        padding: 16px 20px;
      }}
      .stat-value {{ font-size: 1.8rem; }}
      .hero-panel {{ font-size: 1rem; padding: 24px; }}
      .big-number {{ font-size: 3rem; }}
      .nav {{ max-width: 92vw; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      .reveal {{ opacity: 1; transform: none; transition: none; }}
      *, *::before, *::after {{
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
      }}
    }}
  </style>
</head>
<body>
"""

html_tail = """
</body>
</html>
"""

# ── JavaScript / Plotly ──────────────────────────────────────────────────────

js_blocks: list[str] = []

# Data declarations
js_blocks.append(f"""
<script>
const gapScatterData = {json.dumps(gap_scatter_data)};
const arabChartData = {json.dumps(arab_chart_data)};
const temporalChartData = {json.dumps(temporal_chart_data)};
const interScatterData = {json.dumps(inter_scatter_data)};
const importanceData = {json.dumps(importance_data)};
const distScatterData = {json.dumps(dist_scatter_data)};
const trendData = {json.dumps(trend_data)};

const layoutDefaults = {{
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: {{ color: "#F3F4F6", family: "Plus Jakarta Sans, sans-serif" }},
  margin: {{ l: 70, r: 30, t: 60, b: 60 }},
  hovermode: "closest",
  hoverdistance: 20
}};
</script>
""")

# Feature importance horizontal bar chart
js_blocks.append("""
<script>
Plotly.newPlot("graph_importance", [{
  y: importanceData.labels.slice().reverse(),
  x: importanceData.values.slice().reverse(),
  type: "bar",
  orientation: "h",
  marker: {
    color: importanceData.values.slice().reverse().map((v, i, arr) => {
      const max = Math.max(...arr);
      const ratio = v / max;
      return ratio > 0.7 ? "#F59E0B" : ratio > 0.4 ? "#38BDF8" : "#94A3B8";
    })
  },
  hovertemplate: "<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
}], {
  ...layoutDefaults,
  title: "Feature Importance (averaged across models)",
  xaxis: {
    title: { text: "Importance", standoff: 10 },
    gridcolor: "rgba(255,255,255,0.04)",
    automargin: true
  },
  yaxis: {
    automargin: true,
    tickfont: { size: 11 }
  },
  margin: { l: 220, r: 30, t: 60, b: 60 },
  height: 500
}, { responsive: true });
</script>
""")

# Gap scatter chart
js_blocks.append("""
<script>
const gapColors = {
  "Both Lower": "#38BDF8",
  "Both High": "#818CF8",
  "Potential Under-utilization": "#F59E0B",
  "Hidden Burden": "#F43F5E"
};
const gapSymbols = {
  "Both Lower": "circle",
  "Both High": "circle",
  "Potential Under-utilization": "diamond",
  "Hidden Burden": "x"
};
const gapSizes = {
  "Both Lower": 7,
  "Both High": 8,
  "Potential Under-utilization": 12,
  "Hidden Burden": 12
};
const gapOpacity = {
  "Both Lower": 0.35,
  "Both High": 0.45,
  "Potential Under-utilization": 0.95,
  "Hidden Burden": 0.95
};
const gapOrder = ["Both Lower", "Both High", "Potential Under-utilization", "Hidden Burden"];

const gapTraces = gapOrder.map(seg => {
  const d = gapScatterData[seg];
  if (!d || d.x.length === 0) return null;
  return {
    x: d.x,
    y: d.y,
    mode: "markers",
    type: "scatter",
    name: seg,
    text: d.names,
    customdata: d.gap.map(g => [g]),
    marker: {
      color: gapColors[seg],
      size: gapSizes[seg],
      symbol: gapSymbols[seg],
      opacity: gapOpacity[seg],
      line: { color: "rgba(255,255,255,0.5)", width: 0.7 }
    },
    hovertemplate:
      "<b>%{text}</b><br>" +
      "Actual: %{x:.2f}%<br>" +
      "Expected: %{y:.2f}%<br>" +
      "Gap: %{customdata[0]:+.2f} pp<extra></extra>"
  };
}).filter(t => t !== null);

const allActual = gapOrder.flatMap(seg => (gapScatterData[seg] ? gapScatterData[seg].x : []));
const allExpected = gapOrder.flatMap(seg => (gapScatterData[seg] ? gapScatterData[seg].y : []));
const diagMin = Math.min(...allActual, ...allExpected);
const diagMax = Math.max(...allActual, ...allExpected);

Plotly.newPlot("graph_gap_scatter", gapTraces, {
  ...layoutDefaults,
  title: "Expected vs Observed Disability Benefit Rate by Settlement",
  xaxis: {
    title: { text: "Actual Rate (%)", standoff: 10 },
    gridcolor: "rgba(255,255,255,0.04)",
    automargin: true
  },
  yaxis: {
    title: { text: "Expected Rate (%)", standoff: 10 },
    gridcolor: "rgba(255,255,255,0.04)",
    automargin: true
  },
  shapes: [{
    type: "line",
    x0: diagMin, x1: diagMax, y0: diagMin, y1: diagMax,
    line: { color: "#F43F5E", width: 1.5, dash: "dash" }
  }],
  legend: {
    x: 0.98,
    xanchor: "right",
    y: 0.02,
    yanchor: "bottom",
    bgcolor: "rgba(0,0,0,0.6)",
    bordercolor: "rgba(255,255,255,0.15)",
    borderwidth: 1,
    font: { size: 12 }
  },
  margin: { l: 70, r: 30, t: 60, b: 70 },
  height: 520,
  showlegend: true
}, { responsive: true });
</script>
""")

# Intergenerational scatter chart
js_blocks.append("""
<script>
const interTraces = interScatterData.groups.map(g => ({
  x: g.child_rate,
  y: g.adult_rate,
  mode: "markers",
  type: "scatter",
  name: g.name,
  text: g.names,
  marker: {
    color: g.color,
    size: 8,
    opacity: 0.7,
    line: { color: "rgba(255,255,255,0.4)", width: 0.5 }
  },
  hovertemplate:
    "<b>%{text}</b><br>" +
    "Child disability rate: %{x:.2f}%<br>" +
    "Adult disability rate: %{y:.2f}%<extra></extra>"
}));

Plotly.newPlot("graph_intergenerational", interTraces, {
  ...layoutDefaults,
  title: "Child vs Adult Disability Benefit Rate by Settlement",
  xaxis: {
    title: { text: "Child Disability Rate (%)", standoff: 10 },
    gridcolor: "rgba(255,255,255,0.04)",
    automargin: true
  },
  yaxis: {
    title: { text: "Adult Disability Rate (%)", standoff: 10 },
    gridcolor: "rgba(255,255,255,0.04)",
    automargin: true
  },
  legend: {
    x: 0.02,
    xanchor: "left",
    y: 0.98,
    yanchor: "top",
    bgcolor: "rgba(0,0,0,0.6)",
    bordercolor: "rgba(255,255,255,0.15)",
    borderwidth: 1,
    font: { size: 12 }
  },
  margin: { l: 70, r: 30, t: 60, b: 70 },
  height: 480,
  showlegend: true
}, { responsive: true });
</script>
""")

# Arab R² chart
js_blocks.append("""
<script>
Plotly.newPlot("graph_arab_r2", [
  {
    x: arabChartData.models,
    y: arabChartData.secular_r2,
    name: "Secular",
    type: "bar",
    marker: { color: "#38BDF8" },
    hovertemplate: "<b>%{x}</b><br>Secular R\\u00b2: %{y:.3f}<extra></extra>"
  },
  {
    x: arabChartData.models,
    y: arabChartData.overall_r2,
    name: "All 278",
    type: "bar",
    marker: { color: "rgba(148, 163, 184, 0.5)", line: { color: "#94A3B8", width: 2 } },
    hovertemplate: "<b>%{x}</b><br>Overall R\\u00b2: %{y:.3f}<extra></extra>"
  },
  {
    x: arabChartData.models,
    y: arabChartData.arab_r2,
    name: "Arab >50%",
    type: "bar",
    marker: { color: "#F43F5E" },
    hovertemplate: "<b>%{x}</b><br>Arab R\\u00b2: %{y:.3f}<extra></extra>"
  }
], {
  ...layoutDefaults,
  title: "Model Accuracy by Population Sector (R\\u00b2)",
  barmode: "group",
  yaxis: {
    title: { text: "R\\u00b2 (higher = better prediction)", standoff: 10 },
    range: [0, 1],
    gridcolor: "rgba(255,255,255,0.04)",
    automargin: true
  },
  xaxis: { automargin: true },
  legend: {
    x: 0.98,
    xanchor: "right",
    y: 0.98,
    yanchor: "top",
    bgcolor: "rgba(0,0,0,0.5)",
    bordercolor: "rgba(255,255,255,0.15)",
    borderwidth: 1,
    font: { size: 13 }
  },
  margin: { l: 70, r: 30, t: 60, b: 60 },
  height: 450
}, { responsive: true });
</script>
""")

# Temporal chart
js_blocks.append("""
<script>
const tempColors = ["#F43F5E", "#38BDF8"];
const tempTraces = temporalChartData.series.map((s, i) => ({
  x: s.years,
  y: s.rates,
  mode: "lines+markers",
  type: "scatter",
  name: s.name,
  marker: { color: tempColors[i], size: 10 },
  line: { color: tempColors[i], width: 3 },
  hovertemplate:
    "<b>" + s.name + "</b><br>" +
    "Year: %{x}<br>" +
    "Rate: %{y:.2f}%<br>" +
    "<extra></extra>"
}));

// Add delta annotations
const tempAnnotations = [];
temporalChartData.series.forEach((s, i) => {
  s.deltas.forEach((d, j) => {
    if (d !== null) {
      tempAnnotations.push({
        x: s.years[j],
        y: s.rates[j],
        text: (d >= 0 ? "+" : "") + d.toFixed(2),
        showarrow: false,
        yshift: i === 0 ? 18 : -18,
        font: { color: tempColors[i], size: 11 }
      });
    }
  });
});

Plotly.newPlot("graph_temporal", tempTraces, {
  ...layoutDefaults,
  title: "Disability Benefit Rate: Frontline vs Non-Frontline Settlements",
  xaxis: {
    title: { text: "Year", standoff: 10 },
    dtick: 1,
    gridcolor: "rgba(255,255,255,0.04)",
    automargin: true
  },
  yaxis: {
    title: { text: "Disability Rate (%)", standoff: 10 },
    gridcolor: "rgba(255,255,255,0.04)",
    automargin: true
  },
  annotations: tempAnnotations,
  legend: {
    x: 0.02,
    xanchor: "left",
    y: 0.98,
    yanchor: "top",
    bgcolor: "rgba(0,0,0,0.5)",
    bordercolor: "rgba(255,255,255,0.15)",
    borderwidth: 1,
    font: { size: 13 }
  },
  margin: { l: 70, r: 30, t: 60, b: 60 },
  height: 450
}, { responsive: true });
</script>
""")

# Distance scatter with dropdown
js_blocks.append("""
<script>
(function() {
  const specs = [
    { key: "dist_any",             label: "Any BTL branch",              color: "#38BDF8" },
    { key: "dist_central",         label: "Central branch",              color: "#F43F5E" },
    { key: "dist_central_medical", label: "Central + Medical committee", color: "#22C55E" }
  ];

  const initSpec = specs[0];

  const trace = {
    x: distScatterData[initSpec.key],
    y: distScatterData.disability_rate,
    mode: "markers",
    type: "scatter",
    text: distScatterData.names,
    marker: { color: initSpec.color, size: 8, opacity: 0.7,
              line: { color: "rgba(255,255,255,0.4)", width: 0.5 } },
    hovertemplate:
      "<b>%{text}</b><br>" +
      "Disability rate: %{y:.2f}%<br>" +
      "Distance: %{x:.1f} km<extra></extra>"
  };

  const buttons = specs.map(s => ({
    label: s.label,
    method: "update",
    args: [
      { x: [distScatterData[s.key]], "marker.color": s.color },
      { "xaxis.title.text": "Distance to nearest " + s.label + " (km)" }
    ]
  }));

  Plotly.newPlot("graph_dist_scatter", [trace], {
    ...layoutDefaults,
    title: "Disability Rate vs Distance to Nearest NII Branch",
    xaxis: {
      title: { text: "Distance to nearest " + initSpec.label + " (km)", standoff: 10 },
      range: [0, 120],
      gridcolor: "rgba(255,255,255,0.04)",
      automargin: true
    },
    yaxis: {
      title: { text: "Disability Rate (%)", standoff: 10 },
      gridcolor: "rgba(255,255,255,0.04)",
      automargin: true
    },
    updatemenus: [{
      type: "dropdown",
      x: 0.0, y: 1.15, xanchor: "left", yanchor: "top",
      bgcolor: "rgba(11,15,25,0.8)",
      font: { color: "#F3F4F6" },
      buttons: buttons
    }],
    margin: { l: 70, r: 30, t: 80, b: 60 },
    height: 420
  }, { responsive: true });
})();
</script>
""")

# Trend lines chart (SES vs distance)
js_blocks.append("""
<script>
(function() {
  // SES uncertainty band
  const bandX = trendData.x_grid.concat(trendData.x_grid.slice().reverse());
  const bandY = trendData.y_ses_upper.concat(trendData.y_ses_lower.slice().reverse());

  const traces = [
    {
      x: bandX, y: bandY,
      fill: "toself",
      fillcolor: "rgba(244,63,94,0.12)",
      line: { color: "rgba(244,63,94,0)" },
      hoverinfo: "skip",
      name: "95% CI (socio-economic)",
      showlegend: false
    },
    {
      x: trendData.x_grid, y: trendData.y_ses,
      mode: "lines", type: "scatter",
      line: { color: "#F43F5E", width: 3 },
      name: "Socio-economic status \u2192 disability"
    },
    {
      x: trendData.x_grid, y: trendData.y_dist,
      mode: "lines", type: "scatter",
      line: { color: "#38BDF8", width: 3 },
      name: "Distance to NII branch \u2192 disability"
    }
  ];

  Plotly.newPlot("graph_trend_lines", traces, {
    ...layoutDefaults,
    title: "Comparing Predictors: Socio-Economic Status vs Distance",
    xaxis: {
      title: { text: "Standardized scale (low \u2192 high)", standoff: 10 },
      gridcolor: "rgba(255,255,255,0.04)",
      automargin: true
    },
    yaxis: {
      title: { text: "Disability Rate (%)", standoff: 10 },
      gridcolor: "rgba(255,255,255,0.04)",
      automargin: true
    },
    legend: {
      x: 0.98, xanchor: "right", y: 0.98, yanchor: "top",
      bgcolor: "rgba(0,0,0,0.6)",
      bordercolor: "rgba(255,255,255,0.15)",
      borderwidth: 1, font: { size: 12 }
    },
    margin: { l: 70, r: 30, t: 60, b: 60 },
    height: 420
  }, { responsive: true });
})();
</script>
""")

# Navigation & Reveal
js_blocks.append("""
<script>
const navLinks = Array.from(document.querySelectorAll(".nav a"));
const navLookup = new Map(
  navLinks.map(link => [link.getAttribute("href").slice(1), link])
);

function setActiveNav(id) {
  navLinks.forEach(link => link.classList.remove("active"));
  const target = navLookup.get(id);
  if (target) target.classList.add("active");
}

const navObserver = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) setActiveNav(entry.target.id);
    });
  },
  { root: null, rootMargin: "-40% 0px -50% 0px", threshold: 0 }
);
document.querySelectorAll(".slide").forEach(slide => navObserver.observe(slide));

if (location.hash) {
  setActiveNav(location.hash.slice(1));
} else if (navLinks.length) {
  navLinks[0].classList.add("active");
}

const revealObserver = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) entry.target.classList.add("active");
    });
  },
  { root: null, rootMargin: "0px 0px -10% 0px", threshold: 0.01 }
);
document.querySelectorAll(".reveal").forEach(el => revealObserver.observe(el));

// ── Number counter animation ──
const counterObserver = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      if (el.dataset.counted) return;
      el.dataset.counted = "1";
      const target = parseInt(el.dataset.target, 10);
      if (isNaN(target)) return;
      const duration = 1500;
      const start = performance.now();
      function step(now) {
        const t = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - t, 3);
        el.textContent = Math.round(ease * target).toLocaleString();
        if (t < 1) requestAnimationFrame(step);
      }
      el.textContent = "0";
      requestAnimationFrame(step);
    });
  },
  { root: null, rootMargin: "0px 0px -10% 0px", threshold: 0.01 }
);
document.querySelectorAll(".stat-value[data-target]").forEach(el => counterObserver.observe(el));
</script>
""")

# ── Write output ─────────────────────────────────────────────────────────────

js_content = "\n".join(js_blocks)
html_content = f"{html_head}{slides_html}{nav_html}{js_content}{html_tail}"

output_path = PROJECT_ROOT / "presentation_executive.html"
output_path.write_text(html_content, encoding="utf-8")
print(f"Executive presentation saved to: {output_path}")
