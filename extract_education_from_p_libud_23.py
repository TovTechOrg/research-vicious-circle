from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_SHEET_STRIP = "נתונים פיזיים ונתוני אוכלוסייה"
REGIONAL_COUNCIL = "מועצה אזורית"

RECOMMENDED_COLUMNS = [
    "settlement_name",
    "settlement_symbol",
    "district",
    "municipal_status",
    "edu_dropout_pct",
    "edu_bagrut_eligibility_pct",
    "edu_bagrut_uni_req_pct",
    "edu_higher_ed_entry_within_8y_pct",
    "edu_attain_pct_no_info",
    "edu_attain_pct_academic_degree",
    "edu_attain_pct_bagrut_or_higher",
]


def resolve_input_path(cli_path: str | None) -> Path:
    if cli_path:
        path = Path(cli_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Input file not found: {path}")

    preferred = Path("datas_for_research_vicious_circle_project") / "p_libud_23.xlsx"
    fallback = Path("p_libud_23.xlsx")
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "Could not find p_libud_23.xlsx. Tried: "
        f"{preferred.as_posix()} and {fallback.as_posix()}"
    )


def resolve_sheet_name(xl: pd.ExcelFile) -> str:
    for name in xl.sheet_names:
        if name.strip() == TARGET_SHEET_STRIP:
            return name
    if len(xl.sheet_names) > 1:
        return xl.sheet_names[1]
    return xl.sheet_names[0]


def normalize_numeric(series: pd.Series, *, dash_as_zero: bool) -> pd.Series:
    def normalize_value(val: object) -> object:
        if isinstance(val, str):
            token = val.strip()
            if token in ("..", "."):
                return np.nan
            if token == "-":
                return 0 if dash_as_zero else np.nan
            return token
        return val

    s = series.map(normalize_value)
    return pd.to_numeric(s, errors="coerce")


def extract_education(input_path: Path, *, all_columns: bool) -> pd.DataFrame:
    xl = pd.ExcelFile(input_path)
    sheet_name = resolve_sheet_name(xl)

    raw = pd.read_excel(input_path, sheet_name=sheet_name, header=None)
    if raw.shape[0] <= 9:
        raise ValueError(
            f"Sheet {sheet_name!r} has only {raw.shape[0]} rows; cannot apply iloc[9:]."
        )

    data = raw.iloc[9:].copy().reset_index(drop=True)
    before_filter = len(data)
    data = data[data[3] != REGIONAL_COUNCIL]
    after_filter = len(data)

    col_map = {
        0: "settlement_name",
        1: "settlement_symbol",
        2: "district",
        3: "municipal_status",
        166: "edu_dropout_pct",
        169: "edu_bagrut_eligibility_pct",
        170: "edu_bagrut_uni_req_pct",
        172: "edu_higher_ed_entry_within_8y_pct",
        184: "edu_attain_pct_no_info",
        185: "edu_attain_pct_below_elem",
        186: "edu_attain_pct_middle_or_elem_cert",
        187: "edu_attain_pct_highschool_unknown_bagrut",
        188: "edu_attain_pct_highschool_no_bagrut",
        189: "edu_attain_pct_highschool_bagrut",
        190: "edu_attain_pct_postsecondary_nonacademic",
        191: "edu_attain_pct_ba",
        192: "edu_attain_pct_ma",
        193: "edu_attain_pct_phd",
    }

    missing_cols = [idx for idx in col_map if idx >= data.shape[1]]
    if missing_cols:
        raise ValueError(
            "Expected columns are missing in the sheet. "
            f"Missing indices: {missing_cols}. Sheet has {data.shape[1]} columns."
        )

    out = data[list(col_map.keys())].copy()
    out.rename(columns=col_map, inplace=True)

    out["settlement_symbol"] = pd.to_numeric(out["settlement_symbol"], errors="coerce")
    out = out.dropna(subset=["settlement_symbol"]).copy()
    out["settlement_symbol"] = out["settlement_symbol"].astype(int)

    if out["settlement_symbol"].duplicated().any():
        dupes = out.loc[out["settlement_symbol"].duplicated(), "settlement_symbol"].tolist()
        raise ValueError(f"Duplicate settlement_symbol values found: {sorted(set(dupes))[:20]}")

    edu_cols = [
        "edu_dropout_pct",
        "edu_bagrut_eligibility_pct",
        "edu_bagrut_uni_req_pct",
        "edu_higher_ed_entry_within_8y_pct",
        "edu_attain_pct_no_info",
        "edu_attain_pct_below_elem",
        "edu_attain_pct_middle_or_elem_cert",
        "edu_attain_pct_highschool_unknown_bagrut",
        "edu_attain_pct_highschool_no_bagrut",
        "edu_attain_pct_highschool_bagrut",
        "edu_attain_pct_postsecondary_nonacademic",
        "edu_attain_pct_ba",
        "edu_attain_pct_ma",
        "edu_attain_pct_phd",
    ]
    for col in edu_cols:
        out[col] = normalize_numeric(out[col], dash_as_zero=(col == "edu_dropout_pct"))

    out["edu_attain_pct_academic_degree"] = out[
        ["edu_attain_pct_ba", "edu_attain_pct_ma", "edu_attain_pct_phd"]
    ].sum(axis=1, min_count=1)

    out["edu_attain_pct_bagrut_or_higher"] = out[
        [
            "edu_attain_pct_highschool_bagrut",
            "edu_attain_pct_postsecondary_nonacademic",
            "edu_attain_pct_ba",
            "edu_attain_pct_ma",
            "edu_attain_pct_phd",
        ]
    ].sum(axis=1, min_count=1)

    percent_cols = edu_cols + [
        "edu_attain_pct_academic_degree",
        "edu_attain_pct_bagrut_or_higher",
    ]
    out[percent_cols] = out[percent_cols].round(2)

    if all_columns:
        out = out[
            [
                "settlement_name",
                "settlement_symbol",
                "district",
                "municipal_status",
                *edu_cols,
                "edu_attain_pct_academic_degree",
                "edu_attain_pct_bagrut_or_higher",
            ]
        ].copy()
    else:
        out = out[RECOMMENDED_COLUMNS].copy()

    print(f"Loaded rows after iloc[9:]: {before_filter}")
    print(f"Rows after filtering out {REGIONAL_COUNCIL!r}: {after_filter}")
    print(f"Rows in output (valid settlement_symbol): {len(out)}")
    for col in ["edu_bagrut_eligibility_pct", "edu_attain_pct_academic_degree"]:
        if col in out.columns:
            print(f"NaN count {col}: {int(out[col].isna().sum())}")

    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract education indicators from p_libud_23.xlsx into a standalone CSV.",
    )
    parser.add_argument("--input", default=None, help="Path to p_libud_23.xlsx")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (.csv or .xlsx). If omitted, a default name is used.",
    )
    parser.add_argument(
        "--google-sheets",
        action="store_true",
        help="Write a Google Sheets-friendly CSV (semicolon separator, comma decimal).",
    )
    parser.add_argument(
        "--excel",
        action="store_true",
        help="Write an Excel .xlsx file (recommended for Google Sheets uploads).",
    )
    parser.add_argument(
        "--all-columns",
        action="store_true",
        help="Include the full set of education attainment columns in the output.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_path = resolve_input_path(args.input)
    if args.output is None:
        base = "education_from_p_libud_23"
        if args.google_sheets or args.excel:
            base += "_google_sheets"
        if args.all_columns:
            base += "_full"
        ext = ".xlsx" if args.excel else ".csv"
        default_name = base + ext
        output_path = Path("datas_for_research_vicious_circle_project") / default_name
    else:
        output_path = Path(args.output)

    df = extract_education(input_path, all_columns=args.all_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.excel or output_path.suffix.lower() == ".xlsx":
        if output_path.suffix.lower() != ".xlsx":
            output_path = output_path.with_suffix(".xlsx")
        df.to_excel(output_path, index=False)
    else:
        sep = ";" if args.google_sheets else ","
        decimal = "," if args.google_sheets else "."
        df.to_csv(output_path, index=False, encoding="utf-8-sig", sep=sep, decimal=decimal)

    print(f"Wrote: {output_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
