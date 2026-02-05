from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.express as px


DEFAULT_INPUT = Path("datas_for_research_vicious_circle_project") / "education_from_p_libud_23.csv"
DEFAULT_OUTDIR = Path("datas_for_research_vicious_circle_project") / "html_charts" / "education"


@dataclass(frozen=True)
class ChartSpec:
    filename: str
    title: str


def load_education(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path.as_posix()}")

    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df.shape[1] == 1:
            df = pd.read_csv(path, encoding="utf-8-sig", sep=";", decimal=",")
        if df.shape[1] == 1:
            raise ValueError(
                "CSV was not parsed into columns. "
                "Try providing an .xlsx input or a CSV with a recognized separator."
            )

    if "settlement_symbol" in df.columns:
        df["settlement_symbol"] = pd.to_numeric(df["settlement_symbol"], errors="coerce")

    for col in df.columns:
        if col.startswith("edu_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "settlement_name" in df.columns and "settlement_symbol" in df.columns:
        df["label"] = (
            df["settlement_name"].astype(str)
            + " ("
            + df["settlement_symbol"].fillna(-1).astype(int).astype(str)
            + ")"
        )
    else:
        df["label"] = df.index.astype(str)

    return df


def write_html(fig, path: Path, *, title: str) -> None:
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=80, r=30, t=70, b=60),
        font=dict(size=13),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline", full_html=True)


def top_bar(df: pd.DataFrame, metric: str, *, top_n: int, title: str):
    data = df.dropna(subset=[metric]).copy()
    data = data.sort_values(metric, ascending=True).tail(top_n)
    fig = px.bar(
        data,
        x=metric,
        y="label",
        orientation="h",
        color="district" if "district" in data.columns else None,
        hover_data={
            "settlement_symbol": True,
            "district": True if "district" in data.columns else False,
            "municipal_status": True if "municipal_status" in data.columns else False,
        },
    )
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(title=metric)
    return fig


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    title: str,
    color: str | None = None,
    size: str | None = None,
):
    data = df.dropna(subset=[x, y]).copy()
    size_arg = None
    if size and size in data.columns:
        data["__size"] = pd.to_numeric(data[size], errors="coerce").fillna(0).clip(lower=0)
        size_arg = "__size"
    fig = px.scatter(
        data,
        x=x,
        y=y,
        color=color if color in data.columns else None,
        size=size_arg,
        hover_name="settlement_name" if "settlement_name" in data.columns else None,
        hover_data={
            "settlement_symbol": True,
            "district": True if "district" in data.columns else False,
            "municipal_status": True if "municipal_status" in data.columns else False,
            x: True,
            y: True,
        },
    )
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=y)
    return fig


def histogram(df: pd.DataFrame, metric: str, *, title: str):
    data = df.dropna(subset=[metric]).copy()
    fig = px.histogram(data, x=metric, nbins=30)
    fig.update_xaxes(title=metric)
    fig.update_yaxes(title="count")
    return fig


def build_index(outdir: Path, charts: list[ChartSpec]) -> None:
    links = "\n".join(
        f'<li><a href="{spec.filename}">{spec.title}</a></li>' for spec in charts
    )
    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Education charts</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; }}
      h1 {{ margin: 0 0 8px; }}
      p {{ margin: 0 0 16px; color: #333; }}
      ul {{ line-height: 1.7; }}
      code {{ background: #f3f3f3; padding: 2px 5px; border-radius: 4px; }}
    </style>
  </head>
  <body>
    <h1>Education charts</h1>
    <p>Generated from <code>education_from_p_libud_23.csv</code>. Open any chart below:</p>
    <ul>
      {links}
    </ul>
  </body>
</html>
"""
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "index.html").write_text(html, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a few quick charts from the education CSV.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to education CSV/XLSX")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output folder for HTML charts")
    parser.add_argument("--top", type=int, default=25, help="Top N for ranking charts")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)

    df = load_education(input_path)

    charts: list[ChartSpec] = []

    metric = "edu_attain_pct_academic_degree"
    if metric in df.columns:
        spec = ChartSpec("rank_top_academic_degree.html", "Top academic degree share (25–65)")
        fig = top_bar(df, metric, top_n=args.top, title=spec.title)
        write_html(fig, outdir / spec.filename, title=spec.title)
        charts.append(spec)

    metric = "edu_higher_ed_entry_within_8y_pct"
    if metric in df.columns:
        spec = ChartSpec("rank_top_higher_ed_entry.html", "Top higher-ed entry within 8 years (12th grade)")
        fig = top_bar(df, metric, top_n=args.top, title=spec.title)
        write_html(fig, outdir / spec.filename, title=spec.title)
        charts.append(spec)

    metric = "edu_bagrut_uni_req_pct"
    if metric in df.columns:
        spec = ChartSpec("rank_top_bagrut_uni_req.html", "Top Bagrut meeting university requirements (12th grade)")
        fig = top_bar(df, metric, top_n=args.top, title=spec.title)
        write_html(fig, outdir / spec.filename, title=spec.title)
        charts.append(spec)

    if {"edu_bagrut_uni_req_pct", "edu_attain_pct_academic_degree"}.issubset(df.columns):
        spec = ChartSpec(
            "scatter_bagrut_uni_req_vs_academic_degree.html",
            "Bagrut (uni-req) vs academic degree share",
        )
        fig = scatter(
            df,
            x="edu_bagrut_uni_req_pct",
            y="edu_attain_pct_academic_degree",
            title=spec.title,
            color="district",
            size="edu_attain_pct_bagrut_or_higher",
        )
        write_html(fig, outdir / spec.filename, title=spec.title)
        charts.append(spec)

    if {"edu_dropout_pct", "edu_bagrut_uni_req_pct"}.issubset(df.columns):
        spec = ChartSpec(
            "scatter_dropout_vs_bagrut_uni_req.html",
            "Dropout vs Bagrut (uni-req)",
        )
        fig = scatter(
            df,
            x="edu_dropout_pct",
            y="edu_bagrut_uni_req_pct",
            title=spec.title,
            color="district",
            size="edu_attain_pct_no_info",
        )
        write_html(fig, outdir / spec.filename, title=spec.title)
        charts.append(spec)

    metric = "edu_attain_pct_bagrut_or_higher"
    if metric in df.columns:
        spec = ChartSpec("hist_bagrut_or_higher.html", "Distribution: Bagrut or higher (25–65)")
        fig = histogram(df, metric, title=spec.title)
        write_html(fig, outdir / spec.filename, title=spec.title)
        charts.append(spec)

    build_index(outdir, charts)
    print(f"Wrote {len(charts)} charts to: {outdir.as_posix()}")
    print(f"Open: {(outdir / 'index.html').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
