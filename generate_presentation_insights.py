"""
Generate an HTML presentation from research_vicious_circle (8).ipynb.
One chart per slide with notebook-aligned insights and real data.
"""

from __future__ import annotations

import html as html_lib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_PATH = PROJECT_ROOT / "datas_for_research_vicious_circle_project"


def standardize(df: pd.DataFrame) -> np.ndarray:
    arr = df.to_numpy(dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=0)
    std = np.where(std == 0, 1, std)
    return (arr - mean) / std


def minmax_scale(series: pd.Series, feature_range: tuple[float, float] = (-1, 1)) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    if arr.size == 0:
        return series.astype(float)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if not np.isfinite(min_val) or not np.isfinite(max_val) or min_val == max_val:
        return pd.Series(np.full_like(arr, 0.0, dtype=float), index=series.index)
    scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
    scaled = (arr - min_val) * scale + feature_range[0]
    scaled = np.where(np.isfinite(arr), scaled, np.nan)
    return pd.Series(scaled, index=series.index)


def pca_first_component(x: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt.T[:, 0]


def linear_fit(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x_vals = np.asarray(x, dtype=float)
    y_vals = np.asarray(y, dtype=float)
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    if len(x_vals) == 0:
        return 0.0, 0.0
    a = np.vstack([x_vals, np.ones(len(x_vals))]).T
    slope, intercept = np.linalg.lstsq(a, y_vals, rcond=None)[0]
    return float(slope), float(intercept)


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def fmt_num(val: float, digits: int = 2) -> str:
    if pd.isna(val):
        return ""
    return f"{float(val):.{digits}f}"


def fmt_int(val: float) -> str:
    if pd.isna(val):
        return ""
    return f"{int(round(float(val))):,}"


def fmt_cluster(val: float) -> str:
    if pd.isna(val):
        return ""
    return str(int(round(float(val))))


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    head_html = "".join(f"<th>{html_lib.escape(h)}</th>" for h in headers)
    body_html = ""
    for row in rows:
        cells = "".join(f"<td>{html_lib.escape(str(cell))}</td>" for cell in row)
        body_html += f"<tr>{cells}</tr>"
    return (
        "<table class=\"data-table\">"
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
    )


def load_data(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {
        "benefits": pd.read_excel(paths["benefits"], header=None),
        "lamas": pd.read_excel(
            paths["lamas"], sheet_name="נתונים פיזיים ונתוני אוכלוסייה ", header=None
        ),
        "socio_regional": pd.read_excel(paths["socio_regional"], header=None),
        "periph_regional": pd.read_excel(paths["periph_regional"], header=None),
        "coordinates": pd.read_csv(paths["coordinates"]),
    }


def merge_lamas(df_benefits: pd.DataFrame, df_lamas: pd.DataFrame) -> pd.DataFrame:
    before = len(df_benefits)
    df = df_benefits.merge(
        df_lamas[
            [
                "settlement_symbol",
                "socio_economic_index_cluster",
                "socio_economic_index_score",
                "peripherality_index_cluster",
                "peripherality_index_score",
            ]
        ],
        on="settlement_symbol",
        how="left",
    )
    if len(df) != before:
        raise ValueError("Row count mismatch after LAMAS merge.")
    return df


def merge_index_from_regional(
    df_main: pd.DataFrame,
    df_regional: pd.DataFrame,
    index_cols: list[str],
    key: str = "settlement_symbol",
) -> pd.DataFrame:
    before = len(df_main)
    df = df_main.merge(
        df_regional[[key] + index_cols],
        on=key,
        how="left",
        suffixes=("", "_reg"),
    )
    if len(df) != before:
        raise ValueError(f"Row count mismatch after merge on {key}.")
    for col in index_cols:
        df[col] = df[col].combine_first(df[f"{col}_reg"])
    df = df.drop(columns=[f"{col}_reg" for col in index_cols], errors="ignore")
    return df


def merge_coordinates(
    df_main: pd.DataFrame, df_coordinates: pd.DataFrame
) -> pd.DataFrame:
    before = len(df_main)
    df = df_main.merge(
        df_coordinates[["settlement_code", "lat", "lon"]],
        left_on="settlement_symbol",
        right_on="settlement_code",
        how="left",
    ).drop(columns=["settlement_code"], errors="ignore")
    if len(df) != before:
        raise ValueError("Row count mismatch after coordinates merge.")
    return df


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    socio_economic_peripherality_cols = df.columns[-6:-2]
    df = df.dropna(subset=socio_economic_peripherality_cols, how="all")

    cols_to_drop = [
        "settlement_type",
        "injury_allowance",
        "recipients_of_the_senior_citizen_pension_only",
        "recipients_of_the_pension_with_income_supplementation",
        "total_recipients_of_old_age_and_or_survivors_benefits",
        "num_families_receiving_child_benefit",
        "num_children_receiving_child_benefit",
        "families_with_4plus_children_receiving_child_benefit",
        "maternity_benefits",
        "alimony",
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors="ignore")

    categorical_cols = [
        "socio_economic_index_cluster",
        "peripherality_index_cluster",
    ]
    numeric_cols = df.loc[:, "total_population":"unemployment_benefit"].columns
    float_cols = [
        "socio_economic_index_score",
        "peripherality_index_score",
        "lon",
        "lat",
    ]

    for col in df.columns:
        if col in categorical_cols:
            df[col] = df[col].astype("category")
        elif col in numeric_cols:
            df[col] = (
                df[col]
                .astype(str)
                .replace({r"\*\*\*": "5", r"\.\.": "5"}, regex=True)
                .str.replace(",", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif col in float_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        else:
            df[col] = df[col].astype("object")
    return df


print("Loading data...")

paths = {
    "benefits": BASE_PATH / "benefits_2024_12.xlsx",
    "lamas": BASE_PATH / "p_libud_23.xlsx",
    "socio_regional": BASE_PATH / "24_24_230t3.xlsx",
    "periph_regional": BASE_PATH / "24_22_420t3.xlsx",
    "coordinates": BASE_PATH / "israel_settlements_all_with_coords.csv",
}

dfs = load_data(paths)

df_benefits = dfs["benefits"].iloc[5:].copy().reset_index(drop=True)
df_benefits.columns = [
    "settlement_name",
    "settlement_symbol",
    "settlement_type",
    "total_population",
    "population_0_17",
    "population_18_64",
    "population_65_plus",
    "total_recipients_of_old_age_and_or_survivors_benefits",
    "recipients_of_the_pension_with_income_supplementation",
    "recipients_of_the_senior_citizen_pension_only",
    "long_term_care_benefit",
    "general_disability_benefit",
    "special_services_for_persons_with_disabilities",
    "disabled_child_benefit",
    "mobility_benefit",
    "work_injury_victims_receiving_disability_and_dependents_benefits",
    "injury_allowance",
    "num_families_receiving_child_benefit",
    "num_children_receiving_child_benefit",
    "families_with_4plus_children_receiving_child_benefit",
    "maternity_benefits",
    "alimony",
    "income_support_benefit",
    "unemployment_benefit",
]

df_lamas = dfs["lamas"].iloc[9:].copy().reset_index(drop=True)
df_lamas = df_lamas[df_lamas[3] != "מועצה אזורית"]
df_lamas.rename(
    columns={
        1: "settlement_symbol",
        250: "socio_economic_index_cluster",
        251: "socio_economic_index_score",
        256: "peripherality_index_cluster",
        257: "peripherality_index_score",
    },
    inplace=True,
)

df_socio = dfs["socio_regional"].iloc[10:].copy().reset_index(drop=True).iloc[:-8]
df_socio.rename(
    columns={
        5: "settlement_symbol",
        12: "socio_economic_index_cluster",
        10: "socio_economic_index_score",
    },
    inplace=True,
)

df_periph = dfs["periph_regional"].iloc[9:].copy().reset_index(drop=True).iloc[:-4]
df_periph.rename(
    columns={
        4: "settlement_symbol",
        12: "peripherality_index_cluster",
        10: "peripherality_index_score",
    },
    inplace=True,
)

data_master = merge_lamas(df_benefits, df_lamas)
data_master = merge_index_from_regional(
    data_master, df_socio, ["socio_economic_index_cluster", "socio_economic_index_score"]
)
data_master = merge_index_from_regional(
    data_master, df_periph, ["peripherality_index_cluster", "peripherality_index_score"]
)
data_master = merge_coordinates(data_master, dfs["coordinates"])
data_master = clean_values(data_master)

print(f"Loaded {len(data_master)} settlements")

for col in [
    "total_population",
    "population_0_17",
    "population_18_64",
    "population_65_plus",
    "long_term_care_benefit",
    "general_disability_benefit",
    "special_services_for_persons_with_disabilities",
    "disabled_child_benefit",
    "mobility_benefit",
    "income_support_benefit",
    "unemployment_benefit",
    "work_injury_victims_receiving_disability_and_dependents_benefits",
]:
    if col in data_master.columns:
        data_master[col] = pd.to_numeric(data_master[col], errors="coerce").astype(float)

for col in [
    "socio_economic_index_score",
    "peripherality_index_score",
    "lat",
    "lon",
    "socio_economic_index_cluster",
    "peripherality_index_cluster",
]:
    if col in data_master.columns:
        data_master[col] = pd.to_numeric(data_master[col], errors="coerce")

data_master["general_disability_rate"] = (
    data_master["general_disability_benefit"]
    / data_master["population_18_64"]
    * 100
).round(2)
data_master["special_services_disability_rate"] = (
    data_master["special_services_for_persons_with_disabilities"]
    / data_master["population_18_64"]
    * 100
).round(2)
data_master["mobility_disability_rate"] = (
    data_master["mobility_benefit"] / data_master["population_18_64"] * 100
).round(2)
data_master["income_support_rate"] = (
    data_master["income_support_benefit"] / data_master["population_18_64"] * 100
).round(2)
data_master["long_term_care_rate"] = (
    data_master["long_term_care_benefit"] / data_master["population_65_plus"] * 100
).round(2)
data_master["unemployment_rate"] = (
    data_master["unemployment_benefit"] / data_master["population_18_64"] * 100
).round(2)
data_master["work_injury_victims_rate"] = (
    data_master["work_injury_victims_receiving_disability_and_dependents_benefits"]
    / data_master["population_18_64"]
    * 100
).round(2)
data_master["disabled_child_benefit_rate"] = (
    data_master["disabled_child_benefit"] / data_master["population_0_17"] * 100
).round(2)

social_cols = [
    "socio_economic_index_score",
    "peripherality_index_score",
    "income_support_rate",
]
health_cols = [
    "general_disability_rate",
    "special_services_disability_rate",
    "mobility_disability_rate",
]

df_social = data_master[social_cols].copy()
df_social["income_support_rate"] *= -1
df_social_scaled = pd.DataFrame(
    standardize(df_social), columns=social_cols, index=df_social.index
)
raw_social_index = (
    df_social_scaled["socio_economic_index_score"] * 0.5
    + df_social_scaled["peripherality_index_score"] * 0.25
    + df_social_scaled["income_support_rate"] * 0.25
)
data_master["social_index"] = minmax_scale(raw_social_index)

df_health = data_master[health_cols].copy()
for col in health_cols:
    df_health[col] *= -1
df_health_scaled = pd.DataFrame(
    standardize(df_health), columns=health_cols, index=df_health.index
)
raw_health_index = (
    df_health_scaled["general_disability_rate"] * 0.5
    + df_health_scaled["special_services_disability_rate"] * 0.25
    + df_health_scaled["mobility_disability_rate"] * 0.25
)
data_master["health_index"] = minmax_scale(raw_health_index)

df_calc = data_master.copy()
df_calc["socio_economic_index_score"] *= -1
df_calc["peripherality_index_score"] *= -1

cols_for_pca = social_cols + health_cols
df_pca_clean = df_calc.dropna(subset=cols_for_pca).copy()
if len(df_pca_clean) > 0:
    x_health = standardize(df_pca_clean[health_cols])
    x_social = standardize(df_pca_clean[social_cols])

    health_pca_vals = pca_first_component(x_health)
    if np.corrcoef(health_pca_vals, df_pca_clean[health_cols[0]])[0, 1] < 0:
        health_pca_vals = health_pca_vals * -1

    social_pca_vals = pca_first_component(x_social)
    if np.corrcoef(social_pca_vals, df_pca_clean["income_support_rate"])[0, 1] < 0:
        social_pca_vals = social_pca_vals * -1

    data_master.loc[df_pca_clean.index, "health_index_pca"] = health_pca_vals
    data_master.loc[df_pca_clean.index, "social_index_pca"] = social_pca_vals


def get_region(lat_val: float) -> str:
    if pd.isna(lat_val):
        return "Unknown"
    if lat_val > 32.4:
        return "North"
    if lat_val < 31.7:
        return "South"
    return "Center"


data_master["region"] = data_master["lat"].apply(get_region)

print("Preparing graph data...")

# Graph 1: Correlation between social and health indices
df_social_corr = data_master[
    [
        "social_index",
        "health_index",
        "socio_economic_index_cluster",
        "settlement_name",
    ]
].copy()
for col in ["social_index", "health_index", "socio_economic_index_cluster"]:
    df_social_corr[col] = pd.to_numeric(df_social_corr[col], errors="coerce")
df_social_corr = df_social_corr.dropna(
    subset=["social_index", "health_index", "socio_economic_index_cluster"]
)
social_corr_slope, social_corr_intercept = linear_fit(
    df_social_corr["social_index"], df_social_corr["health_index"]
)
social_corr = spearman_corr(
    df_social_corr["social_index"], df_social_corr["health_index"]
)
social_corr_data = {
    "x": df_social_corr["social_index"].round(3).tolist(),
    "y": df_social_corr["health_index"].round(3).tolist(),
    "names": df_social_corr["settlement_name"].tolist(),
    "cluster": df_social_corr["socio_economic_index_cluster"].astype(float).tolist(),
    "slope": social_corr_slope,
    "intercept": social_corr_intercept,
    "corr": social_corr,
}

# Graph 2: Resilience vs Distress (social vs health)
df_res = data_master[
    [
        "social_index",
        "health_index",
        "socio_economic_index_cluster",
        "settlement_name",
        "total_population",
    ]
].copy()
for col in ["social_index", "health_index", "socio_economic_index_cluster", "total_population"]:
    df_res[col] = pd.to_numeric(df_res[col], errors="coerce")
df_res = df_res.dropna(subset=["social_index", "health_index"])
res_slope, res_intercept = linear_fit(df_res["social_index"], df_res["health_index"])
df_res["residual"] = df_res["health_index"] - (
    res_slope * df_res["social_index"] + res_intercept
)
residual_data = {
    "x": df_res["social_index"].round(3).tolist(),
    "y": df_res["health_index"].round(3).tolist(),
    "names": df_res["settlement_name"].tolist(),
    "population": df_res["total_population"].fillna(0).astype(float).tolist(),
    "cluster": df_res["socio_economic_index_cluster"].fillna(0).astype(float).tolist(),
    "residual": df_res["residual"].round(3).tolist(),
    "slope": res_slope,
    "intercept": res_intercept,
}

df_res_table = df_res.dropna(subset=["residual", "total_population"]).copy()
df_res_table = df_res_table[df_res_table["total_population"] > 10_000]

resilient_rows = [
    [
        row["settlement_name"],
        fmt_cluster(row["socio_economic_index_cluster"]),
        fmt_num(row["residual"], 2),
        fmt_int(row["total_population"]),
    ]
    for _, row in df_res_table.sort_values("residual", ascending=False).head(5).iterrows()
]
distress_rows = [
    [
        row["settlement_name"],
        fmt_cluster(row["socio_economic_index_cluster"]),
        fmt_num(row["residual"], 2),
        fmt_int(row["total_population"]),
    ]
    for _, row in df_res_table.sort_values("residual").head(5).iterrows()
]
resilient_table = render_table(
    ["Settlement", "SE Cluster", "Residual", "Population"], resilient_rows
)
distress_table = render_table(
    ["Settlement", "SE Cluster", "Residual", "Population"], distress_rows
)

# Graph 3: Spearman heatmap
corr_matrix = (
    data_master[social_cols + health_cols]
    .corr(method="spearman")
    .loc[health_cols, social_cols]
)

heatmap_data = {
    "z": corr_matrix.values.round(3).tolist(),
    "x": social_cols,
    "y": health_cols,
}

# Graph 4: Intergenerational Trap
df_trap = data_master.copy()
df_trap["socio_economic_index_cluster"] = pd.to_numeric(
    df_trap["socio_economic_index_cluster"], errors="coerce"
)
df_trap = df_trap.dropna(
    subset=[
        "general_disability_rate",
        "disabled_child_benefit_rate",
        "socio_economic_index_cluster",
    ]
)
national_avg_adult = (
    df_trap["general_disability_benefit"].sum()
    / df_trap["population_18_64"].sum()
    * 100
)
national_avg_child = (
    df_trap["disabled_child_benefit"].sum()
    / df_trap["population_0_17"].sum()
    * 100
)
trap_data = {
    "x": df_trap["general_disability_rate"].round(2).tolist(),
    "y": df_trap["disabled_child_benefit_rate"].round(2).tolist(),
    "names": df_trap["settlement_name"].tolist(),
    "cluster": df_trap["socio_economic_index_cluster"].astype(float).tolist(),
    "national_avg_adult": float(national_avg_adult),
    "national_avg_child": float(national_avg_child),
    "x_max": float(df_trap["general_disability_rate"].max()),
    "y_max": float(df_trap["disabled_child_benefit_rate"].max()),
}

high_child = df_trap["disabled_child_benefit_rate"] > df_trap[
    "disabled_child_benefit_rate"
].quantile(0.75)
high_adult = df_trap["general_disability_rate"] > df_trap["general_disability_rate"].quantile(
    0.75
)
trap_top = (
    df_trap[high_child & high_adult][
        [
            "settlement_name",
            "region",
            "socio_economic_index_cluster",
            "disabled_child_benefit_rate",
            "general_disability_rate",
        ]
    ]
    .sort_values("disabled_child_benefit_rate", ascending=False)
    .head(10)
)
trap_rows = [
    [
        row["settlement_name"],
        row["region"],
        fmt_cluster(row["socio_economic_index_cluster"]),
        fmt_num(row["disabled_child_benefit_rate"], 2),
        fmt_num(row["general_disability_rate"], 2),
    ]
    for _, row in trap_top.iterrows()
]
trap_table = render_table(
    ["Settlement", "Region", "SE Cluster", "Child Rate %", "Adult Rate %"], trap_rows
)

# Graph 5: Welfare Trap (chronic vs temporary)
df_ratio = data_master.copy()
df_ratio["chronic_vs_temp_ratio"] = (
    df_ratio["general_disability_benefit"] / df_ratio["unemployment_benefit"]
)
df_ratio = df_ratio[df_ratio["unemployment_benefit"] > 10].dropna(
    subset=["chronic_vs_temp_ratio"]
)
df_ratio = df_ratio.sort_values("chronic_vs_temp_ratio", ascending=False).head(15)
welfare_data = {
    "names": df_ratio["settlement_name"].tolist(),
    "ratio": df_ratio["chronic_vs_temp_ratio"].round(2).tolist(),
}

# Graph 6: Regional comparison (low SE only)
df_low_socio = data_master.copy()
df_low_socio["socio_economic_index_cluster"] = pd.to_numeric(
    df_low_socio["socio_economic_index_cluster"], errors="coerce"
)
df_low_socio = df_low_socio[df_low_socio["socio_economic_index_cluster"] <= 4]
df_low_socio = df_low_socio[df_low_socio["region"] != "Unknown"]
region_data = {}
for region in ["North", "Center", "South"]:
    subset = df_low_socio[df_low_socio["region"] == region]
    region_data[region] = {
        "values": subset["health_index"].dropna().round(3).tolist(),
        "names": subset["settlement_name"].tolist(),
    }

# Graph 7: Disability rates by SE quartile
dfv = data_master.dropna(
    subset=[
        "general_disability_benefit",
        "population_18_64",
        "general_disability_rate",
        "socio_economic_index_score",
        "total_population",
    ]
).copy()
dfv["disability_count"] = dfv["general_disability_benefit"].astype(float)
dfv["pop_working_age"] = dfv["population_18_64"].astype(float)
dfv["disability_rate"] = dfv["general_disability_rate"].astype(float)
dfv["se_score"] = dfv["socio_economic_index_score"].astype(float)
dfv["total_population"] = dfv["total_population"].astype(float)
dfv = dfv[dfv["total_population"] > 10_000].copy()

quartile_labels = ["Q1 (Lowest)", "Q2 (Low)", "Q3 (Medium)", "Q4 (Highest)"]
dfv["se_quartile"] = pd.qcut(dfv["se_score"], 4, labels=quartile_labels)

summary = (
    dfv.groupby("se_quartile", observed=True)
    .agg(avg_disability_rate=("disability_rate", "mean"), n=("disability_rate", "size"))
    .reset_index()
)
summary["se_quartile"] = pd.Categorical(
    summary["se_quartile"], categories=quartile_labels, ordered=True
)
summary = summary.sort_values("se_quartile")

overall_working_age_weighted = (
    dfv["disability_count"].sum() / dfv["pop_working_age"].sum()
) * 100

quartile_data = {
    "labels": summary["se_quartile"].tolist(),
    "values": summary["avg_disability_rate"].round(2).tolist(),
    "counts": summary["n"].astype(int).tolist(),
    "overall_weighted": round(float(overall_working_age_weighted), 2),
}

# Graph 8: General disability vs income support
df_gd_inc = data_master[
    [
        "general_disability_rate",
        "income_support_rate",
        "socio_economic_index_cluster",
        "settlement_name",
    ]
].copy()
for col in ["general_disability_rate", "income_support_rate", "socio_economic_index_cluster"]:
    df_gd_inc[col] = pd.to_numeric(df_gd_inc[col], errors="coerce")
df_gd_inc = df_gd_inc.dropna(
    subset=["general_disability_rate", "income_support_rate", "socio_economic_index_cluster"]
)
gd_inc_slope, gd_inc_intercept = linear_fit(
    df_gd_inc["general_disability_rate"], df_gd_inc["income_support_rate"]
)
gd_inc_corr = spearman_corr(
    df_gd_inc["general_disability_rate"], df_gd_inc["income_support_rate"]
)
gd_inc_data = {
    "x": df_gd_inc["general_disability_rate"].round(2).tolist(),
    "y": df_gd_inc["income_support_rate"].round(2).tolist(),
    "names": df_gd_inc["settlement_name"].tolist(),
    "cluster": df_gd_inc["socio_economic_index_cluster"].astype(float).tolist(),
    "slope": gd_inc_slope,
    "intercept": gd_inc_intercept,
    "corr": gd_inc_corr,
}

# Graph 9: General disability vs SEI
df_gd_sei = data_master[
    [
        "general_disability_rate",
        "socio_economic_index_score",
        "peripherality_index_cluster",
        "socio_economic_index_cluster",
        "settlement_name",
        "total_population",
    ]
].copy()
for col in [
    "general_disability_rate",
    "socio_economic_index_score",
    "peripherality_index_cluster",
    "socio_economic_index_cluster",
    "total_population",
]:
    df_gd_sei[col] = pd.to_numeric(df_gd_sei[col], errors="coerce")
df_gd_sei = df_gd_sei.dropna(
    subset=[
        "general_disability_rate",
        "socio_economic_index_score",
        "peripherality_index_cluster",
    ]
)
gd_sei_slope, gd_sei_intercept = linear_fit(
    df_gd_sei["general_disability_rate"], df_gd_sei["socio_economic_index_score"]
)
gd_sei_corr = spearman_corr(
    df_gd_sei["general_disability_rate"], df_gd_sei["socio_economic_index_score"]
)
gd_sei_data = {
    "x": df_gd_sei["general_disability_rate"].round(2).tolist(),
    "y": df_gd_sei["socio_economic_index_score"].round(2).tolist(),
    "names": df_gd_sei["settlement_name"].tolist(),
    "population": df_gd_sei["total_population"].fillna(0).astype(float).tolist(),
    "periph": df_gd_sei["peripherality_index_cluster"].astype(float).tolist(),
    "periph_min": float(df_gd_sei["peripherality_index_cluster"].min()),
    "periph_max": float(df_gd_sei["peripherality_index_cluster"].max()),
    "slope": gd_sei_slope,
    "intercept": gd_sei_intercept,
    "corr": gd_sei_corr,
}

# Graph 10: Residuals (general disability vs SEI)
df_res_sei = df_gd_sei.copy()
res_sei_slope, res_sei_intercept = linear_fit(
    df_res_sei["general_disability_rate"], df_res_sei["socio_economic_index_score"]
)
df_res_sei["residual"] = df_res_sei["socio_economic_index_score"] - (
    res_sei_slope * df_res_sei["general_disability_rate"] + res_sei_intercept
)
res_sei_data = {
    "x": df_res_sei["general_disability_rate"].round(2).tolist(),
    "y": df_res_sei["socio_economic_index_score"].round(2).tolist(),
    "names": df_res_sei["settlement_name"].tolist(),
    "population": df_res_sei["total_population"].fillna(0).astype(float).tolist(),
    "cluster": df_res_sei["socio_economic_index_cluster"].fillna(0).astype(float).tolist(),
    "residual": df_res_sei["residual"].round(3).tolist(),
    "slope": res_sei_slope,
    "intercept": res_sei_intercept,
}

res_sei_resilient_rows = [
    [
        row["settlement_name"],
        fmt_cluster(row["socio_economic_index_cluster"]),
        fmt_num(row["residual"], 2),
        fmt_int(row["total_population"]),
    ]
    for _, row in df_res_sei.sort_values("residual").head(10).iterrows()
]
res_sei_distress_rows = [
    [
        row["settlement_name"],
        fmt_cluster(row["socio_economic_index_cluster"]),
        fmt_num(row["residual"], 2),
        fmt_int(row["total_population"]),
    ]
    for _, row in df_res_sei.sort_values("residual", ascending=False).head(10).iterrows()
]
res_sei_resilient_table = render_table(
    ["Settlement", "SE Cluster", "Residual", "Population"], res_sei_resilient_rows
)
res_sei_distress_table = render_table(
    ["Settlement", "SE Cluster", "Residual", "Population"], res_sei_distress_rows
)

# Graph 11: LTC vs SEI
df_ltc_sei = data_master[
    [
        "long_term_care_rate",
        "socio_economic_index_score",
        "peripherality_index_cluster",
        "settlement_name",
        "total_population",
    ]
].copy()
for col in [
    "long_term_care_rate",
    "socio_economic_index_score",
    "peripherality_index_cluster",
    "total_population",
]:
    df_ltc_sei[col] = pd.to_numeric(df_ltc_sei[col], errors="coerce")
df_ltc_sei = df_ltc_sei.dropna(
    subset=["long_term_care_rate", "socio_economic_index_score"]
)
ltc_sei_slope, ltc_sei_intercept = linear_fit(
    df_ltc_sei["long_term_care_rate"], df_ltc_sei["socio_economic_index_score"]
)
ltc_sei_corr = spearman_corr(
    df_ltc_sei["long_term_care_rate"], df_ltc_sei["socio_economic_index_score"]
)
ltc_sei_data = {
    "x": df_ltc_sei["long_term_care_rate"].round(2).tolist(),
    "y": df_ltc_sei["socio_economic_index_score"].round(2).tolist(),
    "names": df_ltc_sei["settlement_name"].tolist(),
    "population": df_ltc_sei["total_population"].fillna(0).astype(float).tolist(),
    "periph": df_ltc_sei["peripherality_index_cluster"].fillna(0).astype(float).tolist(),
    "slope": ltc_sei_slope,
    "intercept": ltc_sei_intercept,
    "corr": ltc_sei_corr,
}

# Graph 11b: Senior LTC correlation (not outliers)
df_senior_corr = data_master.copy()
cols_for_senior_corr = [
    "long_term_care_rate",
    "socio_economic_index_score",
    "population_65_plus",
    "socio_economic_index_cluster",
    "settlement_name",
]
for col in [
    "long_term_care_rate",
    "socio_economic_index_score",
    "population_65_plus",
    "socio_economic_index_cluster",
]:
    df_senior_corr[col] = pd.to_numeric(df_senior_corr[col], errors="coerce")

df_senior_corr = df_senior_corr.dropna(subset=cols_for_senior_corr)

ltc_corr_slope, ltc_corr_intercept = linear_fit(
    df_senior_corr["socio_economic_index_score"],
    df_senior_corr["long_term_care_rate"],
)
ltc_corr_corr = spearman_corr(
    df_senior_corr["socio_economic_index_score"],
    df_senior_corr["long_term_care_rate"],
)
ltc_corr_data = {
    "x": df_senior_corr["socio_economic_index_score"].round(2).tolist(),
    "y": df_senior_corr["long_term_care_rate"].round(2).tolist(),
    "names": df_senior_corr["settlement_name"].tolist(),
    "population": df_senior_corr["population_65_plus"].fillna(0).astype(float).tolist(),
    "cluster": df_senior_corr["socio_economic_index_cluster"].fillna(0).astype(float).tolist(),
    "slope": ltc_corr_slope,
    "intercept": ltc_corr_intercept,
    "corr": ltc_corr_corr,
}

# Graph 12: LTC vs income support
df_ltc_inc = data_master[
    [
        "long_term_care_rate",
        "income_support_rate",
        "peripherality_index_cluster",
        "settlement_name",
        "total_population",
    ]
].copy()
for col in [
    "long_term_care_rate",
    "income_support_rate",
    "peripherality_index_cluster",
    "total_population",
]:
    df_ltc_inc[col] = pd.to_numeric(df_ltc_inc[col], errors="coerce")
df_ltc_inc = df_ltc_inc.dropna(subset=["long_term_care_rate", "income_support_rate"])
ltc_inc_slope, ltc_inc_intercept = linear_fit(
    df_ltc_inc["long_term_care_rate"], df_ltc_inc["income_support_rate"]
)
ltc_inc_corr = spearman_corr(
    df_ltc_inc["long_term_care_rate"], df_ltc_inc["income_support_rate"]
)
ltc_inc_data = {
    "x": df_ltc_inc["long_term_care_rate"].round(2).tolist(),
    "y": df_ltc_inc["income_support_rate"].round(2).tolist(),
    "names": df_ltc_inc["settlement_name"].tolist(),
    "population": df_ltc_inc["total_population"].fillna(0).astype(float).tolist(),
    "periph": df_ltc_inc["peripherality_index_cluster"].fillna(0).astype(float).tolist(),
    "slope": ltc_inc_slope,
    "intercept": ltc_inc_intercept,
    "corr": ltc_inc_corr,
}

# Graph 13: Poverty-related disability (methodology scatter)
df_poverty = data_master.copy()
df_poverty["poverty_related_count"] = df_poverty["general_disability_benefit"].fillna(0)
if "long_term_care_benefit" in df_poverty.columns:
    df_poverty["poverty_related_count"] -= df_poverty["long_term_care_benefit"].fillna(0)
if "work_injury_victims_receiving_disability_and_dependents_benefits" in df_poverty.columns:
    df_poverty["poverty_related_count"] -= df_poverty[
        "work_injury_victims_receiving_disability_and_dependents_benefits"
    ].fillna(0)
df_poverty["poverty_related_count"] = df_poverty["poverty_related_count"].clip(lower=0)
df_poverty["poverty_related_rate_per1000_total"] = (
    df_poverty["poverty_related_count"] / df_poverty["total_population"] * 1000
)

df_poverty_plot = df_poverty[
    [
        "socio_economic_index_score",
        "poverty_related_rate_per1000_total",
        "socio_economic_index_cluster",
        "settlement_name",
        "total_population",
    ]
].copy()
for col in [
    "socio_economic_index_score",
    "poverty_related_rate_per1000_total",
    "socio_economic_index_cluster",
    "total_population",
]:
    df_poverty_plot[col] = pd.to_numeric(df_poverty_plot[col], errors="coerce")
df_poverty_plot = df_poverty_plot.dropna(
    subset=["socio_economic_index_score", "poverty_related_rate_per1000_total"]
)
poverty_slope, poverty_intercept = linear_fit(
    df_poverty_plot["socio_economic_index_score"],
    df_poverty_plot["poverty_related_rate_per1000_total"],
)
poverty_corr = spearman_corr(
    df_poverty_plot["socio_economic_index_score"],
    df_poverty_plot["poverty_related_rate_per1000_total"],
)
poverty_data = {
    "x": df_poverty_plot["socio_economic_index_score"].round(2).tolist(),
    "y": df_poverty_plot["poverty_related_rate_per1000_total"].round(2).tolist(),
    "names": df_poverty_plot["settlement_name"].tolist(),
    "population": df_poverty_plot["total_population"].fillna(0).astype(float).tolist(),
    "cluster": df_poverty_plot["socio_economic_index_cluster"].fillna(0).astype(float).tolist(),
    "slope": poverty_slope,
    "intercept": poverty_intercept,
    "corr": poverty_corr,
}

# Graph 14: Poverty-related outliers (general + special services)
data_master["poverty_related_disability"] = (
    data_master["general_disability_benefit"].fillna(0)
    + data_master["special_services_for_persons_with_disabilities"].fillna(0)
)
data_master["poverty_related_rate_per1000"] = (
    data_master["poverty_related_disability"] / data_master["population_18_64"]
) * 1000

df_outliers = data_master[
    [
        "socio_economic_index_score",
        "poverty_related_rate_per1000",
        "socio_economic_index_cluster",
        "settlement_name",
        "total_population",
    ]
].copy()
for col in [
    "socio_economic_index_score",
    "poverty_related_rate_per1000",
    "socio_economic_index_cluster",
    "total_population",
]:
    df_outliers[col] = pd.to_numeric(df_outliers[col], errors="coerce")
df_outliers = df_outliers.dropna(
    subset=["socio_economic_index_score", "poverty_related_rate_per1000"]
)

sei_q1 = df_outliers["socio_economic_index_score"].quantile(0.25)
sei_q3 = df_outliers["socio_economic_index_score"].quantile(0.75)
poverty_rate_q1 = df_outliers["poverty_related_rate_per1000"].quantile(0.25)
poverty_rate_q3 = df_outliers["poverty_related_rate_per1000"].quantile(0.75)

df_outliers["Outlier_Type"] = "Normal"
df_outliers.loc[
    (df_outliers["socio_economic_index_score"] < sei_q1)
    & (df_outliers["poverty_related_rate_per1000"] > poverty_rate_q3),
    "Outlier_Type",
] = "Bad Outlier"
df_outliers.loc[
    (df_outliers["socio_economic_index_score"] < sei_q1)
    & (df_outliers["poverty_related_rate_per1000"] < poverty_rate_q1),
    "Outlier_Type",
] = "Good Outlier"

poverty_outlier_data = {}
for out_type in ["Normal", "Bad Outlier", "Good Outlier"]:
    subset = df_outliers[df_outliers["Outlier_Type"] == out_type]
    poverty_outlier_data[out_type] = {
        "x": subset["socio_economic_index_score"].round(2).tolist(),
        "y": subset["poverty_related_rate_per1000"].round(2).tolist(),
        "names": subset["settlement_name"].tolist(),
        "population": subset["total_population"].fillna(0).astype(float).tolist(),
    }

bad_outliers = (
    df_outliers[df_outliers["Outlier_Type"] == "Bad Outlier"]
    .sort_values(
        ["socio_economic_index_score", "poverty_related_rate_per1000"],
        ascending=[True, False],
    )
    .head(10)
)
good_outliers = (
    df_outliers[df_outliers["Outlier_Type"] == "Good Outlier"]
    .sort_values(
        ["socio_economic_index_score", "poverty_related_rate_per1000"],
        ascending=[True, True],
    )
    .head(10)
)
bad_outlier_rows = [
    [
        row["settlement_name"],
        fmt_num(row["socio_economic_index_score"], 2),
        fmt_num(row["poverty_related_rate_per1000"], 2),
        fmt_int(row["total_population"]),
    ]
    for _, row in bad_outliers.iterrows()
]
good_outlier_rows = [
    [
        row["settlement_name"],
        fmt_num(row["socio_economic_index_score"], 2),
        fmt_num(row["poverty_related_rate_per1000"], 2),
        fmt_int(row["total_population"]),
    ]
    for _, row in good_outliers.iterrows()
]
bad_outlier_table = render_table(
    ["Settlement", "SEI Score", "Rate/1000", "Population"], bad_outlier_rows
)
good_outlier_table = render_table(
    ["Settlement", "SEI Score", "Rate/1000", "Population"], good_outlier_rows
)

# Graph 15: Senior LTC outliers
df_senior_outliers = data_master.copy()
df_senior_outliers["long_term_care_rate"] = (
    df_senior_outliers["long_term_care_benefit"]
    / df_senior_outliers["population_65_plus"]
    * 100
).round(2)
cols_for_senior = [
    "long_term_care_rate",
    "socio_economic_index_score",
    "socio_economic_index_cluster",
    "settlement_name",
    "population_65_plus",
]
for col in cols_for_senior:
    if col != "settlement_name":
        df_senior_outliers[col] = pd.to_numeric(df_senior_outliers[col], errors="coerce")
df_senior_outliers = df_senior_outliers.dropna(subset=cols_for_senior)

sei_q1_senior = df_senior_outliers["socio_economic_index_score"].quantile(0.25)
sei_q3_senior = df_senior_outliers["socio_economic_index_score"].quantile(0.75)
ltc_q1 = df_senior_outliers["long_term_care_rate"].quantile(0.25)
ltc_q3 = df_senior_outliers["long_term_care_rate"].quantile(0.75)

df_senior_outliers["Outlier_Type_Senior"] = "Normal"
df_senior_outliers.loc[
    (df_senior_outliers["socio_economic_index_score"] > sei_q1_senior)
    & (df_senior_outliers["long_term_care_rate"] > ltc_q3),
    "Outlier_Type_Senior",
] = "High Outlier"
df_senior_outliers.loc[
    (df_senior_outliers["socio_economic_index_score"] < sei_q1_senior)
    & (df_senior_outliers["long_term_care_rate"] < ltc_q1),
    "Outlier_Type_Senior",
] = "Low Outlier"

senior_outlier_data = {}
for out_type in ["Normal", "High Outlier", "Low Outlier"]:
    subset = df_senior_outliers[df_senior_outliers["Outlier_Type_Senior"] == out_type]
    senior_outlier_data[out_type] = {
        "x": subset["socio_economic_index_score"].round(2).tolist(),
        "y": subset["long_term_care_rate"].round(2).tolist(),
        "names": subset["settlement_name"].tolist(),
        "population": subset["population_65_plus"].fillna(0).astype(float).tolist(),
    }

high_senior = (
    df_senior_outliers[df_senior_outliers["Outlier_Type_Senior"] == "High Outlier"]
    .sort_values(
        ["socio_economic_index_score", "long_term_care_rate"],
        ascending=[True, False],
    )
    .head(10)
)
low_senior = (
    df_senior_outliers[df_senior_outliers["Outlier_Type_Senior"] == "Low Outlier"]
    .sort_values(
        ["socio_economic_index_score", "long_term_care_rate"],
        ascending=[True, True],
    )
    .head(10)
)
high_senior_rows = [
    [
        row["settlement_name"],
        fmt_num(row["socio_economic_index_score"], 2),
        fmt_num(row["long_term_care_rate"], 2),
        fmt_int(row["population_65_plus"]),
    ]
    for _, row in high_senior.iterrows()
]
low_senior_rows = [
    [
        row["settlement_name"],
        fmt_num(row["socio_economic_index_score"], 2),
        fmt_num(row["long_term_care_rate"], 2),
        fmt_int(row["population_65_plus"]),
    ]
    for _, row in low_senior.iterrows()
]
high_senior_table = render_table(
    ["Settlement", "SEI Score", "LTC Rate %", "Pop 65+"], high_senior_rows
)
low_senior_table = render_table(
    ["Settlement", "SEI Score", "LTC Rate %", "Pop 65+"], low_senior_rows
)


def build_tick_values(min_val: float, max_val: float, count: int = 5) -> tuple[list[float], list[str]]:
    if not np.isfinite(min_val) or not np.isfinite(max_val):
        return [0.0, 1.0], ["0", "1"]
    if min_val == max_val:
        val = float(min_val)
        return [val], [fmt_num(val)]
    ticks = np.linspace(float(min_val), float(max_val), count)
    return [float(t) for t in ticks], [fmt_num(t) for t in ticks]


def mapbox_zoom(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    map_width: int = 980,
    map_height: int = 700,
) -> float:
    if lat_min == lat_max or lon_min == lon_max:
        return 6.5
    lat_min_rad = np.radians(lat_min)
    lat_max_rad = np.radians(lat_max)
    lat_min_merc = np.log(np.tan(np.pi / 4 + lat_min_rad / 2))
    lat_max_merc = np.log(np.tan(np.pi / 4 + lat_max_rad / 2))
    lat_fraction = (lat_max_merc - lat_min_merc) / np.pi
    lon_fraction = (lon_max - lon_min) / 360
    lat_zoom = np.log2(map_height / 256 / lat_fraction)
    lon_zoom = np.log2(map_width / 256 / lon_fraction)
    zoom = float(min(lat_zoom, lon_zoom))
    zoom = zoom + 0.2
    return float(np.clip(zoom, 4.0, 8.5))


# Graph 17: Map data
df_map = data_master.dropna(
    subset=[
        "lat",
        "lon",
        "settlement_name",
        "total_population",
        "socio_economic_index_cluster",
        "general_disability_rate",
        "income_support_rate",
    ]
).copy()
df_map["cluster"] = df_map["socio_economic_index_cluster"].astype(int)
df_map["disability"] = df_map["general_disability_rate"].astype(float)
df_map["income_support"] = df_map["income_support_rate"].astype(float)
df_map["population"] = df_map["total_population"].astype(int)

cluster_norm = ((df_map["cluster"] - 1) / 9).tolist()
green_gold_red = [[0.0, "#D73027"], [0.5, "#C9DD00"], [1.0, "#1A9850"]]
cluster_colors = px.colors.sample_colorscale(green_gold_red, np.linspace(0, 1, 10))
cluster_scale = []
for i, c in enumerate(cluster_colors):
    t = i / 9
    cluster_scale.append([t, c])
    cluster_scale.append([t, c])

cluster_ticks = [i / 9 for i in range(10)]
cluster_ticktext = [str(i) for i in range(1, 11)]
disability_ticks, disability_ticktext = build_tick_values(
    float(df_map["disability"].min()),
    float(df_map["disability"].max()),
)
income_support_ticks, income_support_ticktext = build_tick_values(
    float(df_map["income_support"].min()),
    float(df_map["income_support"].max()),
)
map_data = {
    "lat": df_map["lat"].round(4).tolist(),
    "lon": df_map["lon"].round(4).tolist(),
    "names": df_map["settlement_name"].tolist(),
    "population": df_map["population"].tolist(),
    "cluster": df_map["cluster"].tolist(),
    "cluster_norm": cluster_norm,
    "cluster_scale": cluster_scale,
    "cluster_ticks": cluster_ticks,
    "cluster_ticktext": cluster_ticktext,
    "disability": df_map["disability"].round(2).tolist(),
    "income_support": df_map["income_support"].round(2).tolist(),
    "disability_min": float(df_map["disability"].min()),
    "disability_max": float(df_map["disability"].max()),
    "income_support_min": float(df_map["income_support"].min()),
    "income_support_max": float(df_map["income_support"].max()),
    "pop_max": float(df_map["population"].max()),
    "disability_ticks": disability_ticks,
    "disability_ticktext": disability_ticktext,
    "income_support_ticks": income_support_ticks,
    "income_support_ticktext": income_support_ticktext,
}

lat_min_data = float(df_map["lat"].min())
lat_max_data = float(df_map["lat"].max())
lon_min_data = float(df_map["lon"].min())
lon_max_data = float(df_map["lon"].max())

lat_min = 28.8
lat_max = 33.9
lon_min = 34.2
lon_max = 35.9

lat_pad = (lat_max - lat_min) * 0.005
lon_pad = (lon_max - lon_min) * 0.005
map_data["center_lat"] = (lat_min + lat_max) / 2
map_data["center_lon"] = (lon_min + lon_max) / 2
map_data["zoom"] = mapbox_zoom(lat_min, lat_max, lon_min, lon_max)
map_data["bounds"] = {
    "west": lon_min - lon_pad,
    "east": lon_max + lon_pad,
    "south": lat_min - lat_pad,
    "north": lat_max + lat_pad,
}

intergen_text = """
<div class="insight">
  <h3>Axis Definitions</h3>
  <p>
    The x-axis is the rate of recipients of a general disability pension in each locality, out of the
    entire working-age population in that locality (ages 18-64). The range is from 0 to the settlement
    with the highest general disability rate (14.8%).
  </p>
  <p>
    The y-axis is the rate of recipients of a disabled child benefit in each locality, out of the entire
    population of children in that locality (ages 0 to 17). The range is from 0 to the settlement with
    the highest disabled child benefit rate (10.7%).
  </p>
  <h3>The Data Story</h3>
  <p>
    This scatter plot reveals a <em>possible</em> disturbing correlation between <strong>working-age disability</strong>
    (parents) and <strong>child disability</strong> (future generation). The "Red Zone"
    (top-right quadrant) represents localities where both rates exceed the national weighted average.
  </p>
</div>
"""

welfare_note = """
<div class="warning">
  <h3>Methodological Note: Administrative Data vs. Reality</h3>
  <p><strong>Critical Context for Decision Makers:</strong> The analysis below uses
  <strong>Bituach Leumi administrative data</strong>. It is crucial to distinguish between
  "Benefit Recipients" and the theoretical status of an individual:</p>
  <ul>
    <li><strong>Unemployment Data:</strong> Represents only those currently receiving
      <strong>short-term unemployment benefits</strong>. Long-term unemployed individuals who have exhausted
      their eligibility (common in distressed localities) are <strong>NOT</strong> counted in this denominator.</li>
    <li><strong>Disability Data:</strong> Represents those who successfully claimed benefits.</li>
  </ul>
  <p><strong>Implication:</strong> In the following graph ("The Welfare Trap"), a high ratio often indicates
  <strong>structural, long-term unemployment</strong> where residents have shifted from the temporary
  "unemployment track" to the permanent "disability track".</p>
</div>
"""

welfare_reco = """
<div class="insight">
  <h3>Strategic Recommendations</h3>
  <ol>
    <li><strong>Shift from Placement to Rehabilitation:</strong> Standard employment bureaus are ineffective here.
      Resources should be diverted to <strong>Vocational Rehabilitation Centers</strong> that combine social work
      with soft-skills training.</li>
    <li><strong>"Laron Law" Optimization:</strong> Launch targeted campaigns in these specific towns to educate
      residents on their right to work <em>without</em> losing their entire benefit (incentivizing part-time work).</li>
    <li><strong>Supply-Side Intervention:</strong> Subsidies in these zones should focus exclusively on
      <strong>"Inclusive Employers"</strong> willing to adapt roles for individuals with chronic health issues.</li>
  </ol>
</div>
"""

region_story = """
<div class="insight">
  <h3>The Data Story</h3>
  <p>This analysis compares "apples to apples" (only low socio-economic settlements, clusters 1-4).
  The results are striking: <strong>Geography dictates destiny.</strong></p>
  <ul>
    <li><strong>The North (Blue):</strong> Shows the highest median disability rate (~8%) and the most extreme outliers.</li>
    <li><strong>The Center (Red):</strong> Demonstrates a "protective effect" with significantly lower rates (~5%),
      even for equally poor populations.</li>
  </ul>
  <p><strong>Key Takeaway for Decision Makers:</strong> Poverty in the periphery is "more toxic" than poverty in the
  center. The lack of accessible healthcare and employment infrastructure in the North exacerbates medical conditions.</p>
  <p><strong>Action Item:</strong> <strong>Prioritize the Northern District.</strong> When allocating limited resources
  for health-employment interventions, a "poor town" in the North should receive higher weighting than a "poor town"
  in the Center.</p>
</div>
"""

map_story = """
<div class="insight">
  <h3>Main Findings</h3>
  <p>
    Use the dropdown to compare spatial patterns across socio-economic cluster, disability rate,
    and income support rate. Bubble size reflects population, so dense areas stand out quickly.
  </p>
</div>
"""

quartile_story = """
<div class="insight">
  <h3>Main Findings</h3>
  <p>
    The chart compares the average disability rate across Israeli settlements, grouped into socio-economic
    quartiles—from the lowest (Q1) to the highest (Q4). The analysis includes only settlements with more than
    10,000 residents. Each bar represents the mean disability rate across settlements within that quartile.
    The key takeaway is that there is no linear downward trend across the three lower quartiles (Q1–Q3): the
    values are similar, and Q2 is even slightly higher than Q1. In contrast, the highest quartile (Q4) shows a
    sharp and substantial drop in disability rates compared to the other quartiles.
  </p>
  <p><strong>Benchmark:</strong> Overall weighted disability rate = {overall:.2f}%.</p>
</div>
"""

poverty_outlier_story = """
<div class="insight">
  <h3>Interpretation</h3>
  <p>
    The outlier analysis suggests poverty-related morbidity is shaped not only by socio-economic conditions
    but also by how communities interact with welfare institutions. High outliers reflect contexts where chronic
    poverty and limited labor options translate hardship into formally recognized disability, while low outliers
    likely indicate under-recognition due to access barriers, informal support systems, or cultural resistance to
    registration. Overall, disability rates capture institutional visibility of poverty-related illness as much
    as underlying health.
  </p>
</div>
"""

senior_outlier_story = """
<div class="insight">
  <h3>Interpretation</h3>
  <p>
    Senior long-term care outliers show that while lower socio-economic status is generally associated with higher
    reliance on formal care, deviations reflect institutional and social differences rather than health alone.
    High outliers indicate localities where aging-related dependency is more frequently translated into formal care
    claims, likely due to better access to services, administrative capacity, or reduced reliance on family-based care.
    Low outliers suggest contexts where elder care is absorbed informally within households or communities, leading to
    underutilization of formal care despite similar levels of need. Overall, LTC rates capture patterns of care provision
    and institutional accessibility as much as actual health status.
  </p>
</div>
"""

methodology_text = """
<div class="methodology">
  <h3>Gal's research</h3>
  <p><strong>Research summary: Methodological choices and analytical logic</strong></p>
  <p><strong>Research goal.</strong> The study was designed to examine the widespread claim of a circular relationship
  between poverty, peripherality, and disability, but not only to ask whether a relationship exists - but where it is
  particularly strong, and on which types of disability. The applied goal was to identify focal points where effective
  socio-technological intervention can be focused.</p>

  <p><strong>Choice of the unit of analysis: locality.</strong> The research is conducted at the locality level, not
  the individual, on the assumption that disability is a phenomenon:</p>
  <ul>
    <li>Spatial</li>
    <li>Structural</li>
    <li>Influenced by the socio-economic context</li>
  </ul>
  <p>This choice allows for:</p>
  <ul>
    <li>Integration of socioeconomic, geographic, and health indicators</li>
    <li>Identification of regional patterns and locality anomalies</li>
    <li>Translation of findings into policy and resource allocation</li>
  </ul>

  <p><strong>Why not use the number of benefit recipients?</strong> Absolute numbers are directly affected by the
  size of the locality. Therefore, using them was misleading, artificially strengthens large localities, and prevents
  true comparison between localities.</p>

  <p><strong>The main choice: calculating morbidity rates.</strong></p>
  <div class="formula">Rate = Number of pension recipients / Population of the relevant locality x 1,000</div>
  <p>(or for the entire population, depending on the type of pension)</p>

  <p><strong>Why calculate a rate and not a number?</strong> The transition from using absolute numbers to rates:</p>
  <ul>
    <li>Neutralizes the size of the settlement</li>
    <li>Enables comparison between small and large settlements</li>
    <li>Reveals real structural gaps</li>
  </ul>
  <p>That is: the question is not "where are there more disabled people", but "where is the risk of becoming disabled higher".</p>

  <p><strong>Differentiating between types of disability - a critical decision.</strong> Instead of treating
  "disability" as a single entity, a distinction was made between types of benefits, assuming that each type has a
  different causal mechanism and a different relationship to poverty. Three main categories were examined:</p>
  <ol>
    <li><strong>Work-related injuries</strong> (control group): arise from an occupational event, not the result of a
      cumulative poverty process, expected to be less dependent on socioeconomic status.</li>
    <li><strong>Nursing care benefits (old age)</strong>: highly dependent on age and biological factors, may contaminate
      analysis of poverty-related disability.</li>
    <li><strong>General "poverty-related" disability</strong>: includes general disability, special services, income support,
      and excludes work-related injuries and old age benefits. This represents working-age disability with economic dependence.</li>
  </ol>

  <p><strong>Why a separate calculation by age?</strong> The analysis isolated age groups (mainly 18-64) to avoid biasing
  "old" localities, focus on the population where disability affects labor market participation, and refine the relationship
  between disability, employment, and poverty.</p>

  <p><strong>Examine the relationship to SEI.</strong> After calculating morbidity rates, the relationship between each rate
  and the socioeconomic index (SEI) was examined. Graphs were presented with trend line, dot size by population, coloring by
  SEI cluster, and correlation indices (Pearson, Spearman).</p>

  <p><strong>Why analyze outliers?</strong> Beyond the average, localities that deviate from the general context were identified.
  A qualitative analysis of "good" and "bad" outliers was performed. The goal: to understand which mechanisms strengthen or
  break the connection and to locate intervention and replication centers (POC).</p>

  <p><strong>The main message of the methodology:</strong> The study does not examine "how many disabled people there are", but
  where, in what social context, and with what type of disability - the risk of disability is particularly high. The choice to
  work with rates, differentiate types of benefits, and isolate working ages is what allows us to move from a pretty graph to an
  analysis that can guide action.</p>
</div>
"""

slides: list[str] = []
nav_items: list[tuple[str, str]] = []


def add_slide(slide_id: str, nav_label: str, html_block: str) -> None:
    slides.append(html_block)
    nav_items.append((nav_label, slide_id))


main_contents: list[tuple[str, str]] = [
    ("slide18", "Interactive Map of Israel"),
    ("slide2", "Correlation Between Social-Economic Vulnerability and Disability"),
    ("slide3", "Resilience vs. Distress (Social vs. Health)"),
    ("slide4", "Spearman Correlations: Social vs Health"),
    ("slide5", "The Intergenerational Trap: Child vs. Adult Disability"),
    ("slide13", "Long-Term Care vs Income Support"),
]

appendix_contents: list[tuple[str, str]] = [
    ("slide8", "Disability Rates by Socio-Economic Quartile"),
    ("slide9", "General Disability vs Income Support"),
    ("slide10", "General Disability vs Socio-Economic Index"),
    ("slide11", "Residuals: General Disability vs SEI"),
    ("slide12", "Long-Term Care vs Socio-Economic Index"),
]

def render_toc_list(items: list[tuple[str, str]]) -> str:
    return "".join(
        f'<li><a class="toc-link" href="#{html_lib.escape(slide_id)}">{html_lib.escape(title)}</a></li>'
        for slide_id, title in items
    )

contents_list_html = f"""
<div class="toc-section">
  <h4>Main Slides</h4>
  <ol class="toc-list">{render_toc_list(main_contents)}</ol>
</div>
<div class="toc-section toc-appendix">
  <h4>Appendix (End)</h4>
  <ol class="toc-list">{render_toc_list(appendix_contents)}</ol>
</div>
"""

intro_slide = f"""
<section class="slide" id="slide1">
  <div class="title-block">
    <h1>Vicious Circle Research</h1>
    <p class="subtitle">Socio-economic vulnerability and disability in Israel</p>
    <p class="meta">{len(data_master)} settlements analyzed | Real administrative data</p>
    <div class="intro-lead">
      <p>This project analyzes the factual correlations between Socio-Economic status and medical conditions across Israel, utilizing the most recent data available.We integrated administrative records from the National Insurance Institute (Bituach Leumi)—specifically benefit recipients in localities with over 2,000 residents, current as of December 2024—with official indices from the Central Bureau of Statistics (CBS).  The CBS data includes the Socio-Economic Index (Cluster & Score, updated to 2021) and the Peripherality Index (Cluster & Score, updated to 2020).Based on this unified dataset, we engineered two novel composite features: The Social Index: A weighted combination of the socio-economic score, periphery scale, and income support rates.The Health Index:  An aggregation of general disability, special services for persons with disabilities, and mobility benefits.</p>
    </div>
    <div class="intro-actions">
      <button class="toc-button" type="button" data-toc-open>Open slides list</button>
    </div>
  </div>
  <div class="modal-overlay" id="tocModal" aria-hidden="true">
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="tocTitle">
      <div class="modal-header">
        <h3 id="tocTitle">Slides</h3>
        <button class="modal-close" type="button" data-toc-close aria-label="Close">×</button>
      </div>
      <div class="modal-body">
        <ol class="toc-list">{contents_list_html}</ol>
      </div>
    </div>
  </div>
</section>
<div class="divider"></div>
"""
add_slide("slide1", "Intro", intro_slide)

slide18 = f"""
<section class="slide" id="slide18">
  <h2>Interactive Map of Israel</h2>
  <p class="subtitle">Dropdown: Socio-economic cluster | Disability rate | Income support rate</p>
  {map_story}
  <div class="chart-container map"><div id="graph17_map"></div></div>
</section>
<div class="divider"></div>
"""
add_slide("slide18", "Map", slide18)

slide4 = f"""
<section class="slide" id="slide4">
  <h2>Spearman Correlations: Social vs Health Indicators</h2>
  <p class="subtitle">Negative values mean higher socio-economic score -> lower disability</p>
  <div class="insight">
    <strong>Main Findings:</strong> Values closer to -1 or 1 indicate stronger negative/positive correltation between different indicators.
  </div>
  <div class="chart-container"><div id="graph3_heatmap"></div></div>
</section>
<div class="divider"></div>
"""
add_slide("slide4", "Heatmap", slide4)

slide2 = f"""
<section class="slide" id="slide2">
  <h2>Correlation Between Social-Economic Vulnerability and Disability</h2>
  <p class="subtitle">Social/Health indices | Spearman r = {social_corr_data["corr"]:.2f}</p>
  <div class="insight">
    <h3>Methodology: How We Calculated the Indices</h3>
    <p><strong>Both indices are normalized to a range of -1 (Distress) to +1 (Resilience).</strong></p>
    <h4>1. The Social Index (X-Axis)</h4>
    <p>A weighted measure of a settlement's socio-economic strength:</p>
    <ul>
      <li><strong>50% Socio-Economic Score (CBS):</strong> The official government ranking.</li>
      <li><strong>25% Peripherality Index:</strong> Geographic distance from the center.</li>
      <li><strong>25% Income Support Rate:</strong> A direct measure of deep poverty.</li>
    </ul>
    <p><strong>Meaning:</strong> A score of +1 represents a wealthy, central locality; -1 represents a poor, peripheral one.</p>
    <h4>2. The Health Index (Y-Axis)</h4>
    <p>A weighted measure of the working-age population's health (inverted, so high = healthy):</p>
    <ul>
      <li><strong>50% General Disability Rate:</strong> The volume of disability cases.</li>
      <li><strong>25% Special Services Rate:</strong> Indicates severe disability requiring daily assistance.</li>
      <li><strong>25% Mobility Disability Rate:</strong> Indicates physical mobility limitations.</li>
    </ul>
    <p><strong>Meaning:</strong> A score of +1 represents a locality with very low disability rates (Healthy); -1 represents high disability rates (Unhealthy).</p>
  </div>
  <div class="insight">
    <strong>Main Findings:</strong> Higher social vulnerability aligns with higher disability severity
    across settlements, even after combining multiple indicators into the social and health indices.
  </div>
  <div class="chart-container light"><div id="graph1_pca"></div></div>
</section>
<div class="divider"></div>
"""
add_slide("slide2", "Correlation", slide2)

slide3 = f"""
<section class="slide" id="slide3">
  <h2>Resilience vs. Distress: Deviation from Expected Health Index</h2>
  <p class="subtitle">Residuals from social index -> health index regression</p>
  <div class="insight">
    <h3>Key Anomalies</h3>
    <ul>
      <li><strong>The "Service Magnet" Effect (Regional Hubs):</strong> Cities like Tiberias and Be'er Sheva suffer from "excess disability" (Red Zone). They act as regional hubs, attracting vulnerable populations seeking medical services and public housing. These cities carry a "regional burden".</li>
      <li><strong>Community Resilience:</strong> Settlements like Brukhin, Talmon and Ofra significantly outperform the model. Strong community support networks and younger demographics likely prevent the slide into welfare dependency.</li>
    </ul>
  </div>
  <div class="chart-container"><div id="graph2_residual"></div></div>
  <div class="panel">
    <h3>Top 5 'Distress' Cities (Population > 10K) (Red/Negative Residual)</h3>
    {distress_table}
  </div>
</section>
<div class="divider"></div>
"""
add_slide("slide3", "Residuals", slide3)

slide5 = f"""
<section class="slide" id="slide5">
  <h2>The Intergenerational Trap: Child vs. Adult Disability</h2>
  <p class="subtitle">National averages mark the red zone for priority intervention</p>
  <div class="chart-container"><div id="graph4_intergen"></div></div>
  <div class="two-col">
    <div class="panel">
      {intergen_text}
    </div>
    <div class="panel">
      <h3>Red Zone Priority List (Top 10)</h3>
      {trap_table}
    </div>
  </div>
</section>
<div class="divider"></div>
"""
add_slide("slide5", "Intergen", slide5)

appendix_divider = """
<section class="appendix-divider" id="appendix">
  <h2>Appendix</h2>
</section>
<div class="divider"></div>
"""
slides.append(appendix_divider)

slide8 = f"""
<section class="slide" id="slide8">
  <h2>Disability Rates by Socio-Economic Quartile (Settlements > 10K population)</h2>
  <p class="subtitle">Benchmark line = overall weighted working-age disability rate</p>
  {quartile_story.format(overall=quartile_data["overall_weighted"])}
  <div class="chart-container"><div id="graph7_quartile"></div></div>
</section>
<div class="divider"></div>
"""

slide9 = f"""
<section class="slide" id="slide9">
  <h2>General Disability vs Income Support</h2>
  <p class="subtitle">Spearman r = {gd_inc_data["corr"]:.2f}</p>
  <div class="insight">
    <strong>Main Findings:</strong> Settlements with higher disability rates also tend to show
    higher income support rates, indicating overlapping vulnerability.
  </div>
  <div class="chart-container"><div id="graph8_gd_income"></div></div>
</section>
<div class="divider"></div>
"""

slide10 = f"""
<section class="slide" id="slide10">
  <h2>General Disability vs Socio-Economic Index</h2>
  <p class="subtitle">Spearman r = {gd_sei_data["corr"]:.2f} | Color = peripherality cluster</p>
  <div class="insight">
    <strong>Main Findings:</strong> Lower socio-economic scores are associated with higher disability
    rates, especially in peripheral localities.
  </div>
  <div class="chart-container"><div id="graph9_gd_sei"></div></div>
</section>
<div class="divider"></div>
"""

slide11 = f"""
<section class="slide" id="slide11">
  <h2>Residuals: General Disability vs Socio-Economic Index</h2>
  <p class="subtitle">Deviation from expected SEI given disability rate</p>
  <div class="chart-container"><div id="graph10_residual_sei"></div></div>
</section>
<div class="divider"></div>
"""

slide12 = f"""
<section class="slide" id="slide12">
  <h2>Long-Term Care vs Socio-Economic Index</h2>
  <p class="subtitle">Spearman r = {ltc_sei_data["corr"]:.2f}</p>
  <div class="insight">
    <strong>Main Findings:</strong> LTC rates are influenced by age structure, but still show a
    meaningful association with socio-economic conditions.
  </div>
  <div class="chart-container"><div id="graph11_ltc_sei"></div></div>
</section>
<div class="divider"></div>
"""

slide13 = f"""
<section class="slide" id="slide13">
  <h2>Long-Term Care vs Income Support</h2>
  <p class="subtitle">Spearman r = {ltc_inc_data["corr"]:.2f} | Color = peripherality cluster</p>
  <div class="insight">
    <strong>Main Findings:</strong> Higher income support tends to co-occur with higher LTC rates,
    reflecting overlapping vulnerability among older populations.
  </div>
  <div class="chart-container"><div id="graph12_ltc_income"></div></div>
</section>
<div class="divider"></div>
"""
add_slide("slide13", "LTC vs Income", slide13)
add_slide("slide8", "Quartiles", slide8)
add_slide("slide9", "GD vs Income", slide9)
add_slide("slide10", "GD vs SEI", slide10)
add_slide("slide11", "SEI Residual", slide11)
add_slide("slide12", "LTC vs SEI", slide12)

slides_html = "\n".join(slides)
nav_links = "".join(f"<a href=\"#{sid}\">{html_lib.escape(label)}</a>" for label, sid in nav_items)
nav_html = f"<div class=\"nav\">{nav_links}</div>"

html_head = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Vicious Circle Research - Presentation</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    :root {{
      --bg: #0b141b;
      --bg-2: #102a3a;
      --panel: rgba(255, 255, 255, 0.04);
      --panel-strong: rgba(255, 255, 255, 0.08);
      --accent: #ffb703;
      --accent-2: #2ec4b6;
      --danger: #ff6b6b;
      --text: #e6edf3;
      --muted: #9fb0c2;
    }}
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}
    body {{
      font-family: "IBM Plex Sans", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 18% 10%, rgba(255, 183, 3, 0.15), transparent 42%),
        radial-gradient(circle at 85% 18%, rgba(46, 196, 182, 0.18), transparent 40%),
        linear-gradient(135deg, var(--bg), var(--bg-2));
    }}
    h1, h2, h3, h4 {{
      font-family: "Space Grotesk", sans-serif;
      letter-spacing: 0.2px;
    }}
    h1 {{
      font-size: 2.6rem;
      color: var(--accent);
      text-align: center;
    }}
    h2 {{
      font-size: 1.8rem;
      text-align: center;
      color: #f8f9fb;
    }}
    h3 {{
      font-size: 1.1rem;
      margin-bottom: 8px;
      color: var(--accent-2);
    }}
    h4 {{
      font-size: 1rem;
      margin: 10px 0 6px;
      color: var(--accent);
    }}
    .slide {{
      min-height: 100vh;
      padding: 48px 7vw 90px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 18px;
      animation: fadeInUp 0.6s ease both;
    }}
    .title-block {{
      text-align: center;
      max-width: 900px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .subtitle {{
      font-size: 1.05rem;
      color: var(--muted);
      text-align: center;
    }}
    .appendix-divider {{
      padding: 28px 7vw 22px;
      text-align: center;
    }}
    .appendix-divider h2 {{
      font-size: 2.6rem;
      margin: 0;
      color: var(--accent);
    }}
    .meta {{
      font-size: 0.95rem;
      color: #c5d3e2;
    }}
    .intro-lead {{
      max-width: 1600px;
      margin: 24px auto 0;
      padding: 28px 32px;
      border-radius: 14px;
      border: 1px solid var(--panel-strong);
      background: linear-gradient(135deg, rgba(255, 183, 3, 0.16), rgba(46, 196, 182, 0.08));
      box-shadow: 0 14px 30px rgba(0, 0, 0, 0.2);
    }}
    .intro-lead p {{
      margin: 0;
      font-size: 1.36rem;
      line-height: 1.8;
      font-weight: 500;
      color: #e6edf3;
    }}
    .intro-actions {{
      display: flex;
      justify-content: center;
      margin-top: 18px;
    }}
    .toc-button {{
      appearance: none;
      border: 1px solid rgba(143, 179, 217, 0.35);
      background: rgba(8, 12, 18, 0.55);
      color: #e6edf3;
      padding: 10px 14px;
      border-radius: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: transform 120ms ease, background 120ms ease, border-color 120ms ease;
    }}
    .toc-button:hover {{
      transform: translateY(-1px);
      background: rgba(8, 12, 18, 0.68);
      border-color: rgba(143, 179, 217, 0.55);
    }}
    .toc-link {{
      color: #e6edf3;
      text-decoration: none;
    }}
    .toc-link:hover {{
      text-decoration: underline;
    }}
    .modal-open {{
      overflow: hidden;
    }}
    .modal-overlay {{
      position: fixed;
      inset: 0;
      z-index: 50;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 18px;
      background: rgba(0, 0, 0, 0.6);
      backdrop-filter: blur(6px);
    }}
    .modal-overlay.open {{
      display: flex;
    }}
    .modal {{
      width: min(880px, 96vw);
      max-height: min(78vh, 760px);
      border-radius: 16px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(12, 16, 24, 0.92);
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
      overflow: hidden;
    }}
    .modal-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 16px;
      border-bottom: 1px solid rgba(255,255,255,0.12);
    }}
    .modal-header h3 {{
      margin: 0;
      font-size: 1.05rem;
    }}
    .toc-section {{
      margin-top: 8px;
    }}
    .toc-section h4 {{
      margin: 10px 0 8px;
      font-size: 0.95rem;
      letter-spacing: 0.4px;
      text-transform: uppercase;
      color: #9fb0c2;
    }}
    .toc-appendix h4 {{
      color: #f3d27a;
    }}
    .modal-close {{
      appearance: none;
      border: 0;
      background: transparent;
      color: #e6edf3;
      font-size: 1.4rem;
      line-height: 1;
      cursor: pointer;
      padding: 6px 10px;
      border-radius: 10px;
    }}
    .modal-close:hover {{
      background: rgba(255,255,255,0.08);
    }}
    .modal-body {{
      padding: 10px 18px 18px;
      overflow: auto;
    }}
    .modal-body .toc-list {{
      margin: 0;
      padding-left: 20px;
    }}
    .chart-container {{
      width: 100%;
      max-width: 1350px;
      background: var(--panel);
      border: 1px solid var(--panel-strong);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 18px 40px rgba(0, 0, 0, 0.2);
    }}
    .chart-container.light {{
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.22);
    }}
    .chart-container.map {{
      max-width: 1100px;
      align-self: center;
    }}
    .insight, .warning, .panel, .methodology {{
      width: 100%;
      max-width: 1350px;
      padding: 16px 18px;
      border-radius: 14px;
      border: 1px solid var(--panel-strong);
      background: var(--panel);
      line-height: 1.6;
      font-size: 0.95rem;
    }}
    .insight {{
      border-left: 4px solid var(--accent-2);
      background: rgba(46, 196, 182, 0.08);
    }}
    .warning {{
      border-left: 4px solid var(--danger);
      background: rgba(255, 107, 107, 0.12);
    }}
    .methodology {{
      border-left: 4px solid var(--accent);
      background: rgba(255, 183, 3, 0.08);
      max-height: 340px;
      overflow-y: auto;
    }}
    .formula {{
      margin: 8px 0;
      padding: 8px 10px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      font-family: "IBM Plex Sans", sans-serif;
      font-size: 0.95rem;
    }}
    .two-col {{
      width: 100%;
      max-width: 1350px;
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
    }}
    .two-col .panel {{
      flex: 1;
      min-width: 320px;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    .data-table th, .data-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      text-align: left;
    }}
    .data-table th {{
      color: var(--accent);
      font-weight: 600;
    }}
    .callout {{
      margin-top: 10px;
      font-weight: 600;
      color: var(--accent);
    }}
    .toc-list {{
      margin-left: 20px;
      line-height: 1.7;
      color: var(--muted);
    }}
    .divider {{
      width: 100%;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
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
      background: rgba(8, 12, 18, 0.85);
      padding: 8px 14px;
      border-radius: 999px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      z-index: 1000;
    }}
    .nav a {{
      color: var(--accent);
      text-decoration: none;
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      background: rgba(255, 183, 3, 0.1);
    }}
    .nav a:hover {{
      background: rgba(255, 183, 3, 0.25);
    }}
    .nav a.active {{
      background: var(--accent);
      color: #0b141b;
      font-weight: 700;
      box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.25) inset;
    }}
    @keyframes fadeInUp {{
      from {{
        opacity: 0;
        transform: translateY(12px);
      }}
      to {{
        opacity: 1;
        transform: translateY(0);
      }}
    }}
    @media (max-width: 900px) {{
      .slide {{
        padding: 36px 6vw 90px;
      }}
      h1 {{
        font-size: 2rem;
      }}
      h2 {{
        font-size: 1.4rem;
      }}
      .chart-container {{
        padding: 12px;
      }}
      .nav {{
        max-width: 92vw;
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

js_blocks: list[str] = []
js_blocks.append(f"""
<script>
const socialCorrData = {json.dumps(social_corr_data)};
const residualData = {json.dumps(residual_data)};
const heatmapData = {json.dumps(heatmap_data)};
const trapData = {json.dumps(trap_data)};
const welfareData = {json.dumps(welfare_data)};
const regionData = {json.dumps(region_data)};
const quartileData = {json.dumps(quartile_data)};
const gdIncData = {json.dumps(gd_inc_data)};
const gdSeiData = {json.dumps(gd_sei_data)};
const resSeiData = {json.dumps(res_sei_data)};
const ltcSeiData = {json.dumps(ltc_sei_data)};
const ltcCorrData = {json.dumps(ltc_corr_data)};
const ltcIncData = {json.dumps(ltc_inc_data)};
const povertyData = {json.dumps(poverty_data)};
const povertyOutlierData = {json.dumps(poverty_outlier_data)};
const seniorOutlierData = {json.dumps(senior_outlier_data)};
const mapData = {json.dumps(map_data)};

const layoutDefaults = {{
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: {{ color: "#e6edf3", family: "IBM Plex Sans" }},
  margin: {{ l: 70, r: 30, t: 60, b: 60 }},
  hovermode: "closest",
  hoverdistance: 20
}};

const colorbarOutside = {{
  x: 1.05,
  xanchor: "left",
  len: 0.8,
  thickness: 14,
  tickfont: {{ size: 10 }}
}};

const legendOutside = {{
  x: 1.02,
  y: 1,
  xanchor: "left",
  yanchor: "top",
  bgcolor: "rgba(8,12,18,0.6)",
  bordercolor: "rgba(255,255,255,0.18)",
  borderwidth: 1,
  font: {{ size: 11 }}
}};

function range(arr) {{
  let min = Infinity;
  let max = -Infinity;
  arr.forEach(v => {{
    if (v < min) min = v;
    if (v > max) max = v;
  }});
  return [min, max];
}}

function bubbleSize(val, scale = 3, min = 6, max = 40) {{
  if (!val || val <= 0) return min;
  const size = Math.log(val + 1) * scale;
  return Math.max(min, Math.min(max, size));
}}

function linePoints(xArr, slope, intercept) {{
  const [minX, maxX] = range(xArr);
  const xLine = [minX, maxX];
  const yLine = xLine.map(x => slope * x + intercept);
  return {{ xLine, yLine }};
}}
""")

js_blocks.append("""
// Graph 1: Social vs Health correlation
const socialCorrLine = linePoints(
  socialCorrData.x,
  socialCorrData.slope,
  socialCorrData.intercept
);
Plotly.newPlot("graph1_pca", [
  {
    x: socialCorrData.x,
    y: socialCorrData.y,
    mode: "markers",
    type: "scatter",
    text: socialCorrData.names,
    cliponaxis: true,
    marker: {
      color: socialCorrData.cluster,
      colorscale: "Viridis",
      size: 9,
      showscale: true,
      colorbar: {
        ...colorbarOutside,
        title: { text: "Socio-Economic Index Cluster" },
        len: 0.85
      },
      opacity: 0.85,
      line: { color: "rgba(255,255,255,0.5)", width: 0.7 }
    },
    hovertemplate: "<b>%{text}</b><br>Social Index: %{x:.2f}<br>Health Index: %{y:.2f}<extra></extra>"
  },
  {
    x: socialCorrLine.xLine,
    y: socialCorrLine.yLine,
    mode: "lines",
    type: "scatter",
    line: { color: "darkblue", width: 3 },
    hoverinfo: "skip",
    showlegend: false
  }
], {
  ...layoutDefaults,
  title: "Correlation Between Social-Economic Vulnerability and Disability",
  xaxis: {
    title: { text: "Social-Economic Vulnerability Index (Social Index)", standoff: 10 },
    gridcolor: "rgba(255,255,255,0.08)",
    automargin: true,
    domain: [0, 0.86]
  },
  yaxis: {
    title: { text: "Disability Health Index (Health Index)", standoff: 10 },
    gridcolor: "rgba(255,255,255,0.08)",
    automargin: true
  },
  margin: { l: 70, r: 130, t: 60, b: 70 },
  showlegend: false,
  height: 560
}, { responsive: true });

// Graph 2: Residuals (social vs health)
const residualCustom = residualData.residual.map((r, i) => [
  r,
  residualData.cluster[i],
  residualData.population[i]
]);
const residualLine = linePoints(residualData.x, residualData.slope, residualData.intercept);
const residualColors = [
  [0.0, "#CD2626"],
  [0.5, "#FFC300"],
  [1.0, "#1A9850"]
];
Plotly.newPlot("graph2_residual", [
  {
    x: residualData.x,
    y: residualData.y,
    mode: "markers",
    type: "scatter",
    text: residualData.names,
    customdata: residualCustom,
    cliponaxis: true,
    marker: {
      size: 9,
      color: residualData.residual,
      colorscale: residualColors,
      cmin: -1,
      cmax: 1,
      showscale: true,
      colorbar: {
        ...colorbarOutside,
        title: { text: "", side: "top", font: { size: 12 } },
        len: 0.72,
        y: 0.5,
        yanchor: "middle",
        tickmode: "array",
        tickvals: [-1, 0, 1],
        ticktext: [
          "Distress<br>(Worse than expected)",
          "Normal",
          "Resilient<br>(Better than expected)"
        ]
      },
      opacity: 0.75
    },
    hovertemplate: "<b>%{text}</b><br>Social Index: %{x:.2f}<br>Health Index: %{y:.2f}<br>Residual: %{customdata[0]:.2f}<br>SE Cluster: %{customdata[1]}<extra></extra>"
  },
  {
    x: residualLine.xLine,
    y: residualLine.yLine,
    mode: "lines",
    type: "scatter",
    line: { color: "black", width: 2 },
    hoverinfo: "skip",
    showlegend: false
  }
], {
  ...layoutDefaults,
  title: "Resilience vs. Distress: Deviation from Expected Health Index",
  xaxis: {
    title: "Social Vulnerability (Low = Poor/Peripheral)",
    gridcolor: "rgba(255,255,255,0.08)",
    domain: [0, 0.86]
  },
  yaxis: { title: "Disability Severity (Low = More Disabled)", gridcolor: "rgba(255,255,255,0.08)" },
  annotations: [{
    text: "Anomaly Type",
    xref: "paper",
    yref: "paper",
    x: 1.05,
    y: 0.9,
    xanchor: "left",
    yanchor: "bottom",
    showarrow: false,
    font: { size: 12, color: "#e6edf3" }
  }],
  margin: { l: 70, r: 120, t: 60, b: 70 },
  showlegend: false,
  height: 520
}, { responsive: true });

// Graph 3: Heatmap
const heatmapAnnotations = [];
for (let i = 0; i < heatmapData.y.length; i++) {
  for (let j = 0; j < heatmapData.x.length; j++) {
    const value = heatmapData.z[i][j];
    if (value === null || Number.isNaN(value)) continue;
    heatmapAnnotations.push({
      x: heatmapData.x[j],
      y: heatmapData.y[i],
      text: value.toFixed(2),
      showarrow: false,
      font: { color: "#f8f9fb", size: 12 }
    });
  }
}
Plotly.newPlot("graph3_heatmap", [{
  z: heatmapData.z,
  x: heatmapData.x,
  y: heatmapData.y,
  type: "heatmap",
  colorscale: "RdBu",
  reversescale: true,
  zmin: -1,
  zmax: 1,
  hovertemplate: "%{y} vs %{x}<br>r = %{z:.2f}<extra></extra>"
}], {
  ...layoutDefaults,
  title: "Spearman Correlations: Social vs Health Indicators",
  xaxis: {
    title: "Social / Economic Indicators",
    tickangle: -30,
    tickfont: { size: 11 },
    automargin: true
  },
  yaxis: {
    title: { text: "Health / Disability Indicators", standoff: 18 },
    autorange: "reversed",
    tickfont: { size: 11 },
    automargin: true
  },
  margin: { l: 190, r: 40, t: 60, b: 130 },
  annotations: heatmapAnnotations,
  showlegend: false,
  height: 480
}, { responsive: true });

// Graph 4: Intergenerational trap
Plotly.newPlot("graph4_intergen", [{
  x: trapData.x,
  y: trapData.y,
  mode: "markers",
  type: "scatter",
  text: trapData.names,
  cliponaxis: true,
  marker: {
    size: 9,
    color: trapData.cluster,
    coloraxis: "coloraxis",
    opacity: 0.75
  },
  hovertemplate: "<b>%{text}</b><br>Adult Disability: %{x:.2f}%<br>Child Disability: %{y:.2f}%<extra></extra>"
}], {
  ...layoutDefaults,
  coloraxis: {
    colorscale: "RdYlGn",
    cmin: 1,
    cmax: 10,
    colorbar: {
      ...colorbarOutside,
      title: { text: "SE Cluster" },
      tickmode: "array",
      tickvals: [1,2,3,4,5,6,7,8,9,10],
      ticktext: ["1","2","3","4","5","6","7","8","9","10"]
    }
  },
  title: "The Intergenerational Trap: Child vs. Adult Disability",
  xaxis: {
    title: { text: "Disability Rate (Ages 18-64) %", standoff: 12 },
    gridcolor: "rgba(255,255,255,0.08)",
    automargin: true,
    domain: [0, 0.86]
  },
  yaxis: {
    title: { text: "Child Disability Rate (Ages 0-17) %", standoff: 12 },
    gridcolor: "rgba(255,255,255,0.08)",
    automargin: true
  },
  margin: { l: 80, r: 120, t: 60, b: 90 },
  showlegend: false,
  shapes: [
    {
      type: "line",
      x0: trapData.national_avg_adult,
      x1: trapData.national_avg_adult,
      y0: 0,
      y1: trapData.y_max,
      line: { color: "#f8f9fb", dash: "dash", width: 2 }
    },
    {
      type: "line",
      x0: 0,
      x1: trapData.x_max,
      y0: trapData.national_avg_child,
      y1: trapData.national_avg_child,
      line: { color: "#f8f9fb", dash: "dash", width: 2 }
    },
    {
      type: "rect",
      x0: trapData.national_avg_adult,
      y0: trapData.national_avg_child,
      x1: trapData.x_max,
      y1: trapData.y_max,
      fillcolor: "rgba(255, 59, 48, 0.12)",
      line: { width: 0 }
    }
  ],
  height: 520
}, { responsive: true });

""")

js_blocks.append("""
// Graph 7: SE quartiles
const quartileColors = ["#d73017", "#ff6e31", "#dee00b", "#1ac850"];
Plotly.newPlot("graph7_quartile", [{
  x: quartileData.labels,
  y: quartileData.values,
  type: "bar",
  marker: { color: quartileColors },
  text: quartileData.values.map(v => v.toFixed(2)),
  textposition: "outside",
  customdata: quartileData.counts,
  hovertemplate: "<b>%{x}</b><br>Avg disability rate: %{y:.2f}%<br>Settlements: %{customdata}<extra></extra>"
}], {
  ...layoutDefaults,
  title: "Disability Rates by Socio-Economic Quartile",
  yaxis: { title: "Average Disability Rate (%)", gridcolor: "rgba(255,255,255,0.08)" },
  height: 420,
  showlegend: false,
  shapes: [{
    type: "line",
    x0: -0.5,
    x1: 3.5,
    y0: quartileData.overall_weighted,
    y1: quartileData.overall_weighted,
    line: { color: "#f8f9fb", width: 2, dash: "dash" }
  }],
  annotations: [{
    text: `Overall weighted rate: ${quartileData.overall_weighted.toFixed(2)}%`,
    x: 3.3,
    y: quartileData.overall_weighted + 0.3,
    showarrow: false,
    font: { color: "#f8f9fb", size: 11 }
  }]
}, { responsive: true });

// Graph 8: General disability vs income support
const gdIncLine = linePoints(gdIncData.x, gdIncData.slope, gdIncData.intercept);
Plotly.newPlot("graph8_gd_income", [
  {
    x: gdIncData.x,
    y: gdIncData.y,
    mode: "markers",
    type: "scatter",
    name: "Settlements",
    text: gdIncData.names,
    cliponaxis: true,
    marker: {
      size: 9,
      color: gdIncData.cluster,
      colorscale: "Viridis",
      showscale: true,
      colorbar: { ...colorbarOutside, title: { text: "SE Cluster" } },
      opacity: 0.7
    },
    hovertemplate: "<b>%{text}</b><br>Disability: %{x:.2f}%<br>Income Support: %{y:.2f}%<extra></extra>"
  },
  {
    x: gdIncLine.xLine,
    y: gdIncLine.yLine,
    mode: "lines",
    type: "scatter",
    line: { color: "#ffb703", width: 3 },
    hoverinfo: "skip",
    showlegend: false
  }
], {
  ...layoutDefaults,
  title: "Correlation Between General Disability Rate and Income Support Rate",
  xaxis: {
    title: "General Disability Rate (%)",
    gridcolor: "rgba(255,255,255,0.08)",
    domain: [0, 0.86]
  },
  yaxis: { title: "Income Support Rate (%)", gridcolor: "rgba(255,255,255,0.08)" },
  margin: { l: 70, r: 120, t: 60, b: 70 },
  showlegend: false,
  height: 480
}, { responsive: true });

// Graph 9: General disability vs SEI
const gdSeiLine = linePoints(gdSeiData.x, gdSeiData.slope, gdSeiData.intercept);
Plotly.newPlot("graph9_gd_sei", [
  {
    x: gdSeiData.x,
    y: gdSeiData.y,
    mode: "markers",
    type: "scatter",
    text: gdSeiData.names,
    cliponaxis: true,
    marker: {
      size: 9,
      color: gdSeiData.periph,
      colorscale: "Plasma",
      cmin: gdSeiData.periph_min,
      cmax: gdSeiData.periph_max,
      showscale: true,
      colorbar: { ...colorbarOutside, title: { text: "Peripherality Index" }, x: 1.02 },
      opacity: 0.7
    },
    hovertemplate: "<b>%{text}</b><br>General Disability Rate: %{x:.2f}%<br>Socio-Economic Index Score: %{y:.2f}<extra></extra>"
  },
  {
    x: gdSeiLine.xLine,
    y: gdSeiLine.yLine,
    mode: "lines",
    type: "scatter",
    line: { color: "darkblue", width: 3 },
    hoverinfo: "skip",
    showlegend: false
  }
], {
  ...layoutDefaults,
  title: "Correlation Between General Disability Rate and Socio-Economic Index",
  xaxis: {
    title: "General Disability Rate (%)",
    gridcolor: "rgba(255,255,255,0.08)",
    domain: [0, 0.9]
  },
  yaxis: { title: "Socio-Economic Index Score", gridcolor: "rgba(255,255,255,0.08)" },
  margin: { l: 70, r: 90, t: 60, b: 70 },
  showlegend: false,
  height: 560
}, { responsive: true });

// Graph 10: Residuals (general disability vs SEI)
const resSeiCustom = resSeiData.residual.map((r, i) => [
  r,
  resSeiData.cluster[i],
  resSeiData.population[i]
]);
const resSeiLine = linePoints(resSeiData.x, resSeiData.slope, resSeiData.intercept);
Plotly.newPlot("graph10_residual_sei", [
  {
    x: resSeiData.x,
    y: resSeiData.y,
    mode: "markers",
    type: "scatter",
    text: resSeiData.names,
    customdata: resSeiCustom,
    cliponaxis: true,
    marker: {
      size: 9,
      color: resSeiData.residual,
      colorscale: [
        [0.0, "#CD2626"],
        [0.5, "#FFC300"],
        [1.0, "#1A9850"]
      ],
      cmin: -4,
      cmax: 4,
      showscale: true,
      colorbar: {
        ...colorbarOutside,
        title: { text: "" },
        tickmode: "array",
        tickvals: [-4, 0, 4],
        ticktext: [
          "Resilient<br>(Better than expected)",
          "Normal",
          "Distress<br>(Worse than expected)"
        ]
      },
      opacity: 0.75
    },
    hovertemplate: "<b>%{text}</b><br>General Disability Rate: %{x:.2f}%<br>Socio-Economic Index Score: %{y:.2f}<br>Residual: %{customdata[0]:.2f}<br>SE Cluster: %{customdata[1]}<extra></extra>"
  },
  {
    x: resSeiLine.xLine,
    y: resSeiLine.yLine,
    mode: "lines",
    type: "scatter",
    line: { color: "#f8f9fb", width: 2 },
    hoverinfo: "skip",
    showlegend: false
  }
], {
  ...layoutDefaults,
  title: "Resilience vs. Distress: Deviation from Expected Disability Rate",
  xaxis: {
    title: "General Disability Rate (%)",
    gridcolor: "rgba(255,255,255,0.08)",
    domain: [0, 0.86]
  },
  yaxis: { title: "Socio-Economic Index Score", gridcolor: "rgba(255,255,255,0.08)" },
  annotations: [{
    text: "Anomaly Type",
    xref: "paper",
    yref: "paper",
    x: 1.05,
    y: 0.9,
    xanchor: "left",
    yanchor: "bottom",
    showarrow: false,
    font: { size: 12, color: "#e6edf3" }
  }],
  margin: { l: 70, r: 120, t: 60, b: 70 },
  showlegend: false,
  height: 520
}, { responsive: true });

// Graph 11: LTC vs SEI
const ltcSeiLine = linePoints(ltcSeiData.x, ltcSeiData.slope, ltcSeiData.intercept);
Plotly.newPlot("graph11_ltc_sei", [
  {
    x: ltcSeiData.x,
    y: ltcSeiData.y,
    mode: "markers",
    type: "scatter",
    text: ltcSeiData.names,
    cliponaxis: true,
    marker: {
      size: 9,
      color: ltcSeiData.periph,
      colorscale: "Plasma",
      showscale: true,
      colorbar: { ...colorbarOutside, title: { text: "Peripherality" } },
      opacity: 0.7
    },
    hovertemplate: "<b>%{text}</b><br>LTC Rate: %{x:.2f}%<br>SEI Score: %{y:.2f}<extra></extra>"
  },
  {
    x: ltcSeiLine.xLine,
    y: ltcSeiLine.yLine,
    mode: "lines",
    type: "scatter",
    line: { color: "#ffb703", width: 3 },
    hoverinfo: "skip",
    showlegend: false
  }
], {
  ...layoutDefaults,
  title: "Correlation Between Long-Term Care Rate and Socio-Economic Index",
  xaxis: {
    title: "Long-Term Care Rate (%)",
    gridcolor: "rgba(255,255,255,0.08)",
    domain: [0, 0.86]
  },
  yaxis: { title: "Socio-Economic Index Score", gridcolor: "rgba(255,255,255,0.08)" },
  margin: { l: 70, r: 120, t: 60, b: 70 },
  showlegend: false,
  height: 500
}, { responsive: true });

// Graph 12: LTC vs income support
const ltcIncLine = linePoints(ltcIncData.x, ltcIncData.slope, ltcIncData.intercept);
Plotly.newPlot("graph12_ltc_income", [
  {
    x: ltcIncData.x,
    y: ltcIncData.y,
    mode: "markers",
    type: "scatter",
    text: ltcIncData.names,
    cliponaxis: true,
    marker: {
      size: 9,
      color: ltcIncData.periph,
      colorscale: "Plasma",
      showscale: true,
      colorbar: { ...colorbarOutside, title: { text: "Peripherality" } },
      opacity: 0.7
    },
    hovertemplate: "<b>%{text}</b><br>LTC Rate: %{x:.2f}%<br>Income Support: %{y:.2f}%<extra></extra>"
  },
  {
    x: ltcIncLine.xLine,
    y: ltcIncLine.yLine,
    mode: "lines",
    type: "scatter",
    line: { color: "#ffb703", width: 3 },
    hoverinfo: "skip",
    showlegend: false
  }
], {
  ...layoutDefaults,
  title: "Correlation Between Long-Term Care Rate and Income Support Rate",
  xaxis: {
    title: "Long-Term Care Rate (%)",
    gridcolor: "rgba(255,255,255,0.08)",
    domain: [0, 0.86]
  },
  yaxis: { title: "Income Support Rate (%)", gridcolor: "rgba(255,255,255,0.08)" },
  margin: { l: 70, r: 120, t: 60, b: 70 },
  showlegend: false,
  height: 500
}, { responsive: true });
""")

js_blocks.append("""
// Graph 17: Interactive map
const sizeMax = 38;
const sizeRef = 2.0 * mapData.pop_max / (sizeMax * sizeMax);
const clusterScale = mapData.cluster_scale;

const hoverCluster = "<b>%{text}</b><br>Socio-Economic Cluster: %{customdata[0]}<br>Population: %{customdata[3]:,}<extra></extra>";
const hoverDisability = "<b>%{text}</b><br>General Disability Rate: %{customdata[1]:.2f}%<br>Population: %{customdata[3]:,}<extra></extra>";
const hoverIncomeSupport = "<b>%{text}</b><br>Income Support Rate: %{customdata[2]:.2f}%<br>Population: %{customdata[3]:,}<extra></extra>";

const mapCustom = mapData.lat.map((_, i) => [
  mapData.cluster[i],
  mapData.disability[i],
  mapData.income_support[i],
  mapData.population[i]
]);

Plotly.newPlot("graph17_map", [{
  type: "scattermapbox",
  lat: mapData.lat,
  lon: mapData.lon,
  mode: "markers",
  text: mapData.names,
  customdata: mapCustom,
  hovertemplate: hoverCluster,
  marker: {
    size: mapData.population,
    sizemode: "area",
    sizeref: sizeRef,
    opacity: 0.9,
    color: mapData.cluster_norm,
    colorscale: clusterScale,
    cmin: 0,
    cmax: 1,
    showscale: true,
    colorbar: {
      title: {
        text: "Cluster (1–10)",
        side: "top",
        font: { color: "#e6edf3", size: 12 }
      },
      tickmode: "array",
      tickvals: mapData.cluster_ticks,
      ticktext: mapData.cluster_ticktext,
      tickfont: { color: "#e6edf3", size: 11 },
      len: 0.85,
      x: 1.02,
      xanchor: "left"
    }
  }
}], {
  mapbox: {
    style: "carto-positron",
    center: { lat: mapData.center_lat, lon: mapData.center_lon },
    zoom: mapData.zoom,
    bounds: mapData.bounds
  },
  height: 700,
  margin: { l: 0, r: 140, t: 20, b: 0 },
  paper_bgcolor: "rgba(0,0,0,0)",
  updatemenus: [{
    type: "dropdown",
    direction: "down",
    x: 0.0, y: 1.0,
    xanchor: "left", yanchor: "top",
    showactive: true,
    font: { size: 14 },
    pad: { t: 10, b: 10, r: 10, l: 10 },
    buttons: [
      {
        label: "Socio-Economic Cluster",
        method: "restyle",
        args: [{
          "marker.color": [mapData.cluster_norm],
          "marker.colorscale": [clusterScale],
          "marker.cmin": [0],
          "marker.cmax": [1],
          "marker.showscale": [true],
          "marker.colorbar.title": [{
            text: "Cluster (1–10)",
            side: "top",
            font: { color: "#e6edf3", size: 12 }
          }],
          "marker.colorbar.tickmode": ["array"],
          "marker.colorbar.tickvals": [mapData.cluster_ticks],
          "marker.colorbar.ticktext": [mapData.cluster_ticktext],
          "marker.colorbar.tickfont": [{ color: "#e6edf3", size: 11 }],
          "marker.colorbar.x": [1.02],
          "marker.colorbar.xanchor": ["left"],
          "hovertemplate": [hoverCluster]
        }]
      },
      {
        label: "General Disability Rate",
        method: "restyle",
        args: [{
          "marker.color": [mapData.disability],
          "marker.colorscale": [["0","#1a9850"],["0.45","#fee08b"],["0.75","#d73027"],["1","#000000"]],
          "marker.cmin": [mapData.disability_min],
          "marker.cmax": [mapData.disability_max],
          "marker.showscale": [true],
          "marker.colorbar.title": [{
            text: "General Disability Rate (%)",
            side: "top",
            font: { color: "#e6edf3", size: 12 }
          }],
          "marker.colorbar.tickmode": ["auto"],
          "marker.colorbar.tickfont": [{ color: "#e6edf3", size: 11 }],
          "marker.colorbar.x": [1.02],
          "marker.colorbar.xanchor": ["left"],
          "hovertemplate": [hoverDisability]
        }]
      },
      {
        label: "Income Support Rate",
        method: "restyle",
        args: [{
          "marker.color": [mapData.income_support],
          "marker.colorscale": ["Viridis"],
          "marker.cmin": [mapData.income_support_min],
          "marker.cmax": [4],
          "marker.showscale": [true],
          "marker.colorbar.title": [{
            text: "Income Support Rate (%)",
            side: "top",
            font: { color: "#e6edf3", size: 12 }
          }],
          "marker.colorbar.tickmode": ["auto"],
          "marker.colorbar.tickfont": [{ color: "#e6edf3", size: 11 }],
          "marker.colorbar.x": [1.02],
          "marker.colorbar.xanchor": ["left"],
          "hovertemplate": [hoverIncomeSupport]
        }]
      }
    ]
  }]
}, { responsive: true }).then(gd => {
  const baseZoom = mapData.zoom;
  const baseSizeRef = sizeRef;
  function scaleMapBubbles(zoom) {
    if (zoom === undefined || zoom === null) return;
    const zoomFactor = Math.pow(1.6, zoom - baseZoom);
    const newSizeRef = baseSizeRef / (zoomFactor * zoomFactor);
    Plotly.restyle(gd, { "marker.sizeref": [newSizeRef] });
  }
  scaleMapBubbles(baseZoom);
  gd.on("plotly_relayout", (evt) => {
    if (evt["mapbox.zoom"] !== undefined) {
      scaleMapBubbles(evt["mapbox.zoom"]);
    }
  });
});
</script>
""")

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

const observer = new IntersectionObserver(
  entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        setActiveNav(entry.target.id);
      }
    });
  },
  { root: null, rootMargin: "-35% 0px -55% 0px", threshold: 0 }
);

document.querySelectorAll(".slide").forEach(slide => observer.observe(slide));

if (location.hash) {
  setActiveNav(location.hash.slice(1));
} else if (navLinks.length) {
  navLinks[0].classList.add("active");
}
</script>
""")

js_blocks.append("""
<script>
(function () {
  const openBtn = document.querySelector("[data-toc-open]");
  const overlay = document.getElementById("tocModal");
  if (!openBtn || !overlay) return;

  const closeBtn = overlay.querySelector("[data-toc-close]");
  let lastFocus = null;

  function openModal() {
    lastFocus = document.activeElement;
    overlay.classList.add("open");
    overlay.setAttribute("aria-hidden", "false");
    document.body.classList.add("modal-open");
    if (closeBtn) setTimeout(() => closeBtn.focus(), 0);
  }

  function closeModal() {
    overlay.classList.remove("open");
    overlay.setAttribute("aria-hidden", "true");
    document.body.classList.remove("modal-open");
    if (lastFocus && typeof lastFocus.focus === "function") lastFocus.focus();
  }

  openBtn.addEventListener("click", openModal);
  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeModal();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && overlay.classList.contains("open")) closeModal();
  });

  overlay.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener("click", () => closeModal());
  });
})();
</script>
""")

js_content = "\n".join(js_blocks)
html_content = f"{html_head}{slides_html}{nav_html}{js_content}{html_tail}"

output_path = PROJECT_ROOT / "presentation_main.html"
output_path.write_text(html_content, encoding="utf-8")
print(f"Presentation saved to: {output_path}")
