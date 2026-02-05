from __future__ import annotations

"""
Build the project's "master" settlement-level dataset (benefits + CBS indices + geography + demographics).

This script extracts/cleans multiple raw inputs from `datas_for_research_vicious_circle_project/` and produces a single
table that can be used by notebooks/presentations.

Primary use (from a notebook):

    from build_master_dataset import build_master_dataset
    df = build_master_dataset(save=True)

CLI:
    python build_master_dataset.py
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_data_dir(cli_data_dir: str | None) -> Path:
    if cli_data_dir:
        p = Path(cli_data_dir)
        if not p.exists():
            raise FileNotFoundError(f"Data dir not found: {p.as_posix()}")
        return p

    preferred = PROJECT_ROOT / "datas_for_research_vicious_circle_project"
    if preferred.exists():
        return preferred
    return PROJECT_ROOT


def default_paths(data_dir: Path) -> dict[str, Path]:
    return {
        "benefits": data_dir / "benefits_2024_12.xlsx",
        "lamas": data_dir / "p_libud_23.xlsx",
        "socio_regional": data_dir / "24_24_230t3.xlsx",
        "periph_regional": data_dir / "24_22_420t3.xlsx",
        "coordinates": data_dir / "israel_settlements_all_with_coords.csv",
        "settlements2022": data_dir / "bycode2022.xlsx",
        "haredi_population": data_dir / "The_Haredi_population.xlsx",
        "haredi_population2020": data_dir / "haredi_local_authorities_economic_development_lamas_fixed.xlsx",
        "average_salary": data_dir / "average_monthly_salary.xlsx",
    }


def require_path(path: Path, *, key: str) -> None:
    if path.exists():
        return
    raise FileNotFoundError(f"Missing input for {key!r}: {path.as_posix()}")


def load_benefits(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    df = raw.iloc[5:].copy().reset_index(drop=True)
    df.columns = [
        "settlement_name",
        "settlement_symbol",
        "settlement_type",
        "total_population",
        "population_0_17",
        "population_18_64",
        "population_65_plus",
        "total_recipients_of_old_age_and/or_survivors’_benefits",
        "recipients_of_the_pension_with_income_supplementation",
        "recipients_of_the_senior_citizen_pension_only",
        "long_term_care_benefit",
        "general_disability_benefit",
        "special_services_for_persons_with_disabilities",
        "disabled_child_benefit",
        "mobility_benefit",
        "work_injury_victims_receiving_disability_and_dependents’_benefits",
        "injury_allowance",
        "num_families_receiving_child_benefit",
        "num_children_receiving_child_benefit",
        "families_with_4+_children_receiving_child_benefit",
        "maternity_benefits",
        "alimony",
        "income_support_benefit",
        "unemployment_benefit",
    ]
    df["settlement_name"] = df["settlement_name"].astype(str).str.strip()
    return df


def load_lamas(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="נתונים פיזיים ונתוני אוכלוסייה ", header=None)
    df = raw.iloc[9:].copy().reset_index(drop=True)
    df = df[df[3] != "מועצה אזורית"].copy()
    df.rename(
        columns={
            1: "settlement_symbol",
            250: "socio_economic_index_cluster",
            251: "socio_economic_index_score",
            256: "peripherality_index_cluster",
            257: "peripherality_index_score",
        },
        inplace=True,
    )
    return df


def load_socio_regional(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    df = raw.iloc[10:].copy().reset_index(drop=True)
    df = df.iloc[:-8].copy()
    df.rename(
        columns={
            5: "settlement_symbol",
            12: "socio_economic_index_cluster",
            10: "socio_economic_index_score",
        },
        inplace=True,
    )
    return df


def load_periph_regional(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    df = raw.iloc[9:].copy().reset_index(drop=True)
    df = df.iloc[:-4].copy()
    df.rename(
        columns={
            4: "settlement_symbol",
            12: "peripherality_index_cluster",
            10: "peripherality_index_score",
        },
        inplace=True,
    )
    return df


def load_coordinates(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_settlements2022(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["jewish_population_percentage"] = round(
        (df["מזה: יהודים"] / df["סך הכל אוכלוסייה 2022"]) * 100,
        2,
    )
    df["arab_population_percentage"] = round(
        (df["ערבים"] / df["סך הכל אוכלוסייה 2022"]) * 100,
        2,
    )
    df.rename(columns={"סמל יישוב": "settlement_symbol", "שם מחוז": "district_name"}, inplace=True)
    return df


def load_haredi_2023(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    df = raw.iloc[6:].copy().reset_index(drop=True)
    df = df.iloc[:-5].copy()
    df.rename(columns={1: "settlement_name", 3: "haredi_population_percentage"}, inplace=True)
    df["settlement_name"] = df["settlement_name"].astype(str).str.strip()
    df["haredi_population_percentage"] = df["haredi_population_percentage"] * 100
    return df


def load_haredi_2020(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.copy()
    df["estimated_haredi_percent_2023"] = df["haredi_percent"] * 1.136
    return df


def load_average_salary(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(
        columns={
            "יישוב": "settlement_name",
            "מספר עובדים 2023": "num_workers_2023",
            "ממוצע לחודש עבודה 2023": "average_monthly_salary_2023",
        }
    )
    df = df[~df["settlement_name"].str.contains("סה\"כ", na=False)].copy()
    df["settlement_name"] = df["settlement_name"].astype(str).str.strip()
    return df


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
    assert len(df) == before, f"Lost rows in LAMAS merge (before={before}, after={len(df)})"
    return df


def merge_index_from_regional(
    df_main: pd.DataFrame,
    df_regional: pd.DataFrame,
    index_cols: list[str],
    *,
    key: str = "settlement_symbol",
) -> pd.DataFrame:
    before = len(df_main)
    df = df_main.merge(df_regional[[key] + index_cols], on=key, how="left", suffixes=("", "_reg"))
    assert len(df) == before, (
        f"Row count mismatch after merge on {key!r} (before={before}, after={len(df)})"
    )
    for col in index_cols:
        df[col] = df[col].combine_first(df[f"{col}_reg"])
    df = df.drop(columns=[f"{col}_reg" for col in index_cols])
    return df


def merge_coordinates(df_main: pd.DataFrame, df_coordinates: pd.DataFrame) -> pd.DataFrame:
    before = len(df_main)
    df = df_main.merge(
        df_coordinates[["settlement_code", "lat", "lon"]],
        left_on="settlement_symbol",
        right_on="settlement_code",
        how="left",
    )
    df = df.drop(columns=["settlement_code"])
    assert len(df) == before, f"Lost rows in COORDINATES merge (before={before}, after={len(df)})"
    return df


def merge_settlements2022(df_main: pd.DataFrame, df_settlements2022: pd.DataFrame) -> pd.DataFrame:
    before = len(df_main)
    df = df_main.merge(
        df_settlements2022[
            [
                "settlement_symbol",
                "district_name",
                "jewish_population_percentage",
                "arab_population_percentage",
            ]
        ],
        on="settlement_symbol",
        how="left",
    )
    assert len(df) == before, (
        f"Lost rows in settlements2022 merge (before={before}, after={len(df)})"
    )
    return df


def merge_haredi_2023(df_main: pd.DataFrame, df_haredi: pd.DataFrame) -> pd.DataFrame:
    before = len(df_main)
    df = df_main.merge(
        df_haredi[["settlement_name", "haredi_population_percentage"]],
        on="settlement_name",
        how="left",
    )
    assert len(df) == before, f"Lost rows in haredi(2023) merge (before={before}, after={len(df)})"
    return df


def merge_haredi_2020(df_main: pd.DataFrame, df_haredi_2020: pd.DataFrame) -> pd.DataFrame:
    before = len(df_main)
    df = df_main.merge(
        df_haredi_2020[["locality_code", "estimated_haredi_percent_2023"]],
        left_on="settlement_symbol",
        right_on="locality_code",
        how="left",
    )
    assert len(df) == before, f"Lost rows in haredi(2020) merge (before={before}, after={len(df)})"
    df = df.drop(columns=["locality_code"])
    df["haredi_population_percentage"] = df["haredi_population_percentage"].combine_first(
        df["estimated_haredi_percent_2023"]
    )
    df = df.drop(columns=["estimated_haredi_percent_2023"])
    return df


def merge_average_salary(df_main: pd.DataFrame, df_salary: pd.DataFrame) -> pd.DataFrame:
    before = len(df_main)
    df = df_main.merge(
        df_salary[["settlement_name", "num_workers_2023", "average_monthly_salary_2023"]],
        on="settlement_name",
        how="left",
    )
    assert len(df) == before, (
        f"Lost rows in average_salary merge (before={before}, after={len(df)})"
    )
    return df


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(
        subset=[
            "socio_economic_index_cluster",
            "socio_economic_index_score",
            "peripherality_index_cluster",
            "peripherality_index_score",
        ],
        how="all",
    )

    cols_to_drop = [
        "settlement_type",
        "injury_allowance",
        "recipients_of_the_senior_citizen_pension_only",
        "recipients_of_the_pension_with_income_supplementation",
        "total_recipients_of_old_age_and/or_survivors’_benefits",
        "num_families_receiving_child_benefit",
        "num_children_receiving_child_benefit",
        "families_with_4+_children_receiving_child_benefit",
        "maternity_benefits",
        "alimony",
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    categorial_cols = ["socio_economic_index_cluster", "peripherality_index_cluster"]
    numeric_cols = df.loc[:, "total_population":"unemployment_benefit"].columns.append(
        pd.Index(["num_workers_2023"])
    )
    float_cols = [
        "socio_economic_index_score",
        "peripherality_index_score",
        "lon",
        "lat",
        "jewish_population_percentage",
        "arab_population_percentage",
        "haredi_population_percentage",
        "jewish_non_haredi_population_percentage",
        "average_monthly_salary_2023",
    ]
    percentage_cols = ["jewish_population_percentage", "arab_population_percentage", "haredi_population_percentage"]

    for col in df.columns:
        if col in categorial_cols:
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

    for col in percentage_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if "jewish_population_percentage" in df.columns and "haredi_population_percentage" in df.columns:
        df["jewish_non_haredi_population_percentage"] = (
            df["jewish_population_percentage"] - df["haredi_population_percentage"]
        )
    return df


def save_dataset(
    df: pd.DataFrame,
    name: str,
    output_dir: Path,
    *,
    write_csv: bool,
    write_pkl: bool,
    write_excel: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_csv:
        csv_path = output_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ Saved CSV: {csv_path.as_posix()}")
    if write_pkl:
        pkl_path = output_dir / f"{name}.pkl"
        df.to_pickle(pkl_path)
        print(f"✅ Saved Pickle: {pkl_path.as_posix()}")
    if write_excel:
        xlsx_path = output_dir / f"{name}.xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f"✅ Saved Excel: {xlsx_path.as_posix()}")


def build_master_dataset(
    *,
    data_dir: str | Path | None = None,
    paths: dict[str, str | Path] | None = None,
    output_dir: str | Path | None = None,
    name: str = "benefits_final",
    save: bool = True,
    verbose: bool = True,
    write_csv: bool = True,
    write_pkl: bool = True,
    write_excel: bool = False,
) -> pd.DataFrame:
    """
    Build the master dataset and optionally save it as CSV+PKL.
    """

    resolved_data_dir = resolve_data_dir(str(data_dir) if data_dir is not None else None)
    base_paths = default_paths(resolved_data_dir)
    if paths:
        for k, v in paths.items():
            candidate = Path(v)
            if candidate.exists():
                base_paths[k] = candidate
                continue
            if not candidate.is_absolute():
                nested = resolved_data_dir / candidate
                if nested.exists():
                    base_paths[k] = nested
                    continue
            base_paths[k] = candidate

    required_keys = [
        "benefits",
        "lamas",
        "socio_regional",
        "periph_regional",
        "coordinates",
        "settlements2022",
        "haredi_population",
        "haredi_population2020",
        "average_salary",
    ]
    for key in required_keys:
        require_path(base_paths[key], key=key)

    resolved_output_dir = Path(output_dir) if output_dir else resolved_data_dir / "data" / "processed"

    if verbose:
        print(f"📦 Data dir: {resolved_data_dir.as_posix()}")
        print(f"📝 Output dir: {resolved_output_dir.as_posix()}")

    df_benefits = load_benefits(base_paths["benefits"])
    df_lamas = load_lamas(base_paths["lamas"])
    df_socio = load_socio_regional(base_paths["socio_regional"])
    df_periph = load_periph_regional(base_paths["periph_regional"])
    df_coordinates = load_coordinates(base_paths["coordinates"])
    df_settlements2022 = load_settlements2022(base_paths["settlements2022"])
    df_haredi_2023 = load_haredi_2023(base_paths["haredi_population"])
    df_haredi_2020 = load_haredi_2020(base_paths["haredi_population2020"])
    df_salary = load_average_salary(base_paths["average_salary"])

    data_master = merge_lamas(df_benefits, df_lamas)
    data_master = merge_index_from_regional(
        data_master,
        df_socio,
        index_cols=["socio_economic_index_cluster", "socio_economic_index_score"],
    )
    data_master = merge_index_from_regional(
        data_master,
        df_periph,
        index_cols=["peripherality_index_cluster", "peripherality_index_score"],
    )
    data_master = merge_coordinates(data_master, df_coordinates)
    data_master = merge_settlements2022(data_master, df_settlements2022)
    data_master = merge_haredi_2023(data_master, df_haredi_2023)
    data_master = merge_haredi_2020(data_master, df_haredi_2020)
    data_master = merge_average_salary(data_master, df_salary)
    data_master = clean_values(data_master)

    if verbose:
        print(f"✅ Built master dataset: {len(data_master)} rows × {data_master.shape[1]} cols")

    if save:
        save_dataset(
            data_master,
            name=name,
            output_dir=resolved_output_dir,
            write_csv=write_csv,
            write_pkl=write_pkl,
            write_excel=write_excel,
        )

    return data_master


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the master dataset (benefits + indices + demographics).")
    p.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing raw input files (default: datas_for_research_vicious_circle_project/ if present).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to save outputs (default: <data-dir>/data/processed).",
    )
    p.add_argument("--name", default="benefits_final", help="Output base name (default: benefits_final)")
    p.add_argument("--no-save", action="store_true", help="Do not write CSV/PKL; just build in-memory.")
    p.add_argument("--excel", action="store_true", help="Also write an Excel .xlsx file.")
    p.add_argument("--no-pkl", action="store_true", help="Do not write the .pkl output.")
    p.add_argument("--benefits", default=None, help="Override path to benefits Excel (benefits_2024_12.xlsx)")
    p.add_argument("--lamas", default=None, help="Override path to LAMAS Excel (p_libud_23.xlsx)")
    p.add_argument(
        "--socio-regional",
        dest="socio_regional",
        default=None,
        help="Override path to socio-economic regional Excel (24_24_230t3.xlsx)",
    )
    p.add_argument(
        "--periph-regional",
        dest="periph_regional",
        default=None,
        help="Override path to peripherality regional Excel (24_22_420t3.xlsx)",
    )
    p.add_argument(
        "--coordinates",
        default=None,
        help="Override path to coordinates CSV (israel_settlements_all_with_coords.csv)",
    )
    p.add_argument(
        "--settlements2022",
        default=None,
        help="Override path to settlements 2022 Excel (bycode2022.xlsx)",
    )
    p.add_argument(
        "--haredi-population",
        dest="haredi_population",
        default=None,
        help="Override path to haredi population (2023) Excel (The_Haredi_population.xlsx)",
    )
    p.add_argument(
        "--haredi-population2020",
        dest="haredi_population2020",
        default=None,
        help="Override path to haredi population (2020) Excel (haredi_local_authorities_economic_development_lamas_fixed.xlsx)",
    )
    p.add_argument(
        "--average-salary",
        dest="average_salary",
        default=None,
        help="Override path to average salary Excel (average_monthly_salary.xlsx)",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    overrides: dict[str, str] = {}
    for key in [
        "benefits",
        "lamas",
        "socio_regional",
        "periph_regional",
        "coordinates",
        "settlements2022",
        "haredi_population",
        "haredi_population2020",
        "average_salary",
    ]:
        val = getattr(args, key, None)
        if val:
            overrides[key] = val
    build_master_dataset(
        data_dir=args.data_dir,
        paths=overrides or None,
        output_dir=args.output_dir,
        name=args.name,
        save=not args.no_save,
        verbose=True,
        write_excel=bool(args.excel),
        write_pkl=not bool(args.no_pkl),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
