"""
Generate an interactive HTML presentation for the education dataset extracted from p_libud_23.xlsx.

Input:  datas_for_research_vicious_circle_project/education_from_p_libud_23.csv (default)
Output: presentation_education.html (default)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_INPUT_CANDIDATES = [
    PROJECT_ROOT / "datas_for_research_vicious_circle_project" / "education_from_p_libud_23.csv",
    PROJECT_ROOT / "datas_for_research_vicious_circle_project" / "education_from_p_libud_23_google_sheets.xlsx",
    PROJECT_ROOT / "datas_for_research_vicious_circle_project" / "education_from_p_libud_23_google_sheets.csv",
]


def resolve_input_path(cli_path: str | None) -> Path:
    if cli_path:
        path = Path(cli_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Input not found: {path.as_posix()}")

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find an education dataset. Tried:\n"
        + "\n".join(f"- {p.as_posix()}" for p in DEFAULT_INPUT_CANDIDATES)
    )


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df.shape[1] == 1:
            df = pd.read_csv(path, encoding="utf-8-sig", sep=";", decimal=",")

    expected = [
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
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            "Input dataset is missing expected columns:\n"
            + "\n".join(f"- {c}" for c in missing)
        )

    df = df.copy()
    df["settlement_symbol"] = pd.to_numeric(df["settlement_symbol"], errors="coerce")
    for col in expected:
        if col.startswith("edu_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["settlement_symbol"]).copy()
    df["settlement_symbol"] = df["settlement_symbol"].astype(int)
    df["settlement_name"] = df["settlement_name"].astype(str)
    df["district"] = df["district"].astype(str)
    df["municipal_status"] = df["municipal_status"].astype(str)
    return df


def safe_records(df: pd.DataFrame, cols: list[str]) -> list[dict[str, object]]:
    tmp = df[cols].copy()
    tmp = tmp.replace({np.nan: None})
    return tmp.to_dict(orient="records")


def spearman_corr(df: pd.DataFrame, cols: list[str]) -> list[list[float | None]]:
    corr = df[cols].corr(method="spearman")
    out: list[list[float | None]] = []
    for _, row in corr.iterrows():
        out.append([None if pd.isna(v) else float(v) for v in row.tolist()])
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an education-only HTML presentation (Plotly).")
    parser.add_argument("--input", default=None, help="Path to education CSV/XLSX")
    parser.add_argument(
        "--output",
        default="presentation_education.html",
        help="Output HTML path (default: presentation_education.html)",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    input_path = resolve_input_path(args.input)
    output_path = Path(args.output)

    df = load_dataset(input_path)

    metric_specs = [
        {
            "key": "edu_attain_pct_academic_degree",
            "short": "Academic degree (25–65)",
            "label": "Academic degree share (BA+MA+PhD), ages 25–65 (%)",
            "higher_is_better": True,
        },
        {
            "key": "edu_attain_pct_bagrut_or_higher",
            "short": "Bagrut+ (25–65)",
            "label": "Bagrut or higher, ages 25–65 (%)",
            "higher_is_better": True,
        },
        {
            "key": "edu_higher_ed_entry_within_8y_pct",
            "short": "Higher-ed entry (8y)",
            "label": "Entered higher education within 8 years (12th grade cohort) (%)",
            "higher_is_better": True,
        },
        {
            "key": "edu_bagrut_uni_req_pct",
            "short": "Bagrut (uni-req)",
            "label": "Bagrut meeting university requirements (12th grade cohort) (%)",
            "higher_is_better": True,
        },
        {
            "key": "edu_bagrut_eligibility_pct",
            "short": "Bagrut eligibility",
            "label": "Bagrut eligibility (12th grade cohort) (%)",
            "higher_is_better": True,
        },
        {
            "key": "edu_dropout_pct",
            "short": "Dropout",
            "label": "Student dropout rate (%)",
            "higher_is_better": False,
        },
        {
            "key": "edu_attain_pct_no_info",
            "short": "No info (25–65)",
            "label": "No education info (ages 25–65) (%)",
            "higher_is_better": False,
        },
    ]

    corr_cols = [
        "edu_attain_pct_academic_degree",
        "edu_attain_pct_bagrut_or_higher",
        "edu_higher_ed_entry_within_8y_pct",
        "edu_bagrut_uni_req_pct",
        "edu_dropout_pct",
        "edu_attain_pct_no_info",
    ]

    records = safe_records(
        df,
        [
            "settlement_name",
            "settlement_symbol",
            "district",
            "municipal_status",
            *[m["key"] for m in metric_specs],
        ],
    )

    corr_matrix = spearman_corr(df, corr_cols)

    summary = {
        "n_settlements": int(df.shape[0]),
        "n_districts": int(df["district"].nunique()),
        "districts": sorted(df["district"].dropna().unique().tolist()),
        "input_file": None,
    }
    try:
        summary["input_file"] = input_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        summary["input_file"] = input_path.name

    data_json = json.dumps(records, ensure_ascii=False)
    metrics_json = json.dumps(metric_specs, ensure_ascii=False)
    summary_json = json.dumps(summary, ensure_ascii=False)
    corr_json = json.dumps(
        {"cols": corr_cols, "matrix": corr_matrix},
        ensure_ascii=False,
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Education Snapshot — Interactive Presentation</title>
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
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: "IBM Plex Sans", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 18% 10%, rgba(255, 183, 3, 0.15), transparent 42%),
        radial-gradient(circle at 85% 18%, rgba(46, 196, 182, 0.18), transparent 40%),
        linear-gradient(135deg, var(--bg), var(--bg-2));
    }}
    h1, h2, h3 {{ font-family: "Space Grotesk", sans-serif; letter-spacing: 0.2px; }}
    h1 {{ font-size: 2.5rem; color: var(--accent); text-align: center; }}
    h2 {{ font-size: 1.7rem; text-align: center; color: #f8f9fb; }}
    h3 {{ font-size: 1.05rem; color: var(--accent-2); }}
    .slide {{
      min-height: 100vh;
      padding: 48px 7vw 90px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 18px;
      animation: fadeInUp 0.6s ease both;
    }}
    .subtitle {{
      font-size: 1.05rem;
      color: var(--muted);
      text-align: center;
      max-width: 980px;
      line-height: 1.5;
    }}
    .grid {{
      width: min(1200px, 100%);
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 14px;
      padding: 14px 16px;
      backdrop-filter: blur(8px);
    }}
    .card p {{ color: var(--muted); line-height: 1.45; }}
    .chart-container {{
      width: min(1200px, 100%);
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 16px;
      padding: 14px;
      backdrop-filter: blur(8px);
    }}
    .table-container {{
      width: min(1200px, 100%);
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 16px;
      padding: 12px 14px;
      backdrop-filter: blur(8px);
    }}
    .table-scroll {{
      max-height: 520px;
      overflow: auto;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.10);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: rgba(8, 12, 18, 0.92);
      z-index: 2;
      font-weight: 600;
      color: var(--text);
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      font-size: 0.92rem;
      color: var(--muted);
      text-overflow: ellipsis;
      overflow: hidden;
      white-space: nowrap;
    }}
    td strong {{ color: var(--text); font-weight: 600; }}
    tr:hover td {{
      background: rgba(255,255,255,0.05);
      color: var(--text);
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.85rem;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(8, 12, 18, 0.55);
      color: var(--text);
    }}
    .pill.missing {{
      border-color: rgba(255,107,107,0.5);
      color: var(--danger);
    }}
    .pill.ok {{
      border-color: rgba(46,196,182,0.45);
      color: var(--accent-2);
    }}
    .controls {{
      width: min(1200px, 100%);
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: center;
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 14px;
      padding: 12px 14px;
    }}
    .controls label {{
      display: flex;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .controls select, .controls input {{
      background: rgba(8, 12, 18, 0.6);
      color: var(--text);
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 10px;
      padding: 8px 10px;
      outline: none;
    }}
    .controls input[type="text"] {{
      min-width: 240px;
    }}
    .controls input[type="range"] {{ padding: 0; }}
    .divider {{
      height: 1px;
      width: 100%;
      background: rgba(255,255,255,0.08);
    }}
    .nav {{
      position: fixed;
      left: 50%;
      transform: translateX(-50%);
      bottom: 18px;
      display: flex;
      gap: 10px;
      padding: 10px 12px;
      background: rgba(8,12,18,0.72);
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 999px;
      backdrop-filter: blur(10px);
      z-index: 100;
      flex-wrap: wrap;
      justify-content: center;
      max-width: min(1100px, 92vw);
    }}
    .nav a {{
      color: var(--muted);
      text-decoration: none;
      font-size: 0.9rem;
      padding: 6px 10px;
      border-radius: 999px;
      transition: background 0.2s ease, color 0.2s ease;
    }}
    .nav a:hover {{ background: rgba(255,255,255,0.08); color: var(--text); }}
    .nav a.active {{ background: rgba(255,183,3,0.18); color: var(--accent); }}
    @keyframes fadeInUp {{
      from {{ opacity: 0; transform: translateY(10px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>

<section class="slide" id="slide1">
  <h1>Education Snapshot</h1>
  <p class="subtitle">
    Interactive exploration of education indicators by settlement (CBS/LAMAS).
    Use the controls to rank settlements, explore relationships, and see distributions.
  </p>
  <div class="grid">
    <div class="card">
      <h3>Dataset</h3>
      <p id="summary_box"></p>
    </div>
    <div class="card">
      <h3>What this is</h3>
      <p>
        This presentation is based on <code>p_libud_23.xlsx</code> (sheet:
        <code>נתונים פיזיים ונתוני אוכלוסייה</code>), exported into
        <code>education_from_p_libud_23.csv</code>.
        Values are percentages unless stated otherwise.
      </p>
    </div>
  </div>
  <div class="chart-container"><div id="chart_corr"></div></div>
</section>
<div class="divider"></div>

<section class="slide" id="slide2">
  <h2>Rank settlements (interactive)</h2>
  <p class="subtitle">Choose a metric, pick “best” or “worst”, and adjust Top N.</p>
  <div class="controls">
    <label>Metric
      <select id="rank_metric"></select>
    </label>
    <label>Mode
      <select id="rank_mode">
        <option value="best">Best</option>
        <option value="worst">Worst</option>
      </select>
    </label>
    <label>Top N
      <input id="rank_n" type="range" min="5" max="60" step="1" value="25" />
      <span id="rank_n_label">25</span>
    </label>
  </div>
  <div class="chart-container"><div id="chart_rank"></div></div>
</section>
<div class="divider"></div>

<section class="slide" id="slide3">
  <h2>Explore relationships (interactive)</h2>
  <p class="subtitle">Scatter plot by district; marker size can reflect “no info”.</p>
  <div class="controls">
    <label>X
      <select id="scatter_x"></select>
    </label>
    <label>Y
      <select id="scatter_y"></select>
    </label>
    <label>Size
      <select id="scatter_size">
        <option value="">None</option>
        <option value="edu_attain_pct_no_info">No info (25–65)</option>
      </select>
    </label>
  </div>
  <p class="subtitle" id="scatter_stats"></p>
  <div class="chart-container"><div id="chart_scatter"></div></div>
</section>
<div class="divider"></div>

<section class="slide" id="slide4">
  <h2>Distributions (interactive)</h2>
  <p class="subtitle">How do settlements distribute across each metric?</p>
  <div class="controls">
    <label>Metric
      <select id="hist_metric"></select>
    </label>
  </div>
  <div class="chart-container"><div id="chart_hist"></div></div>
</section>
<div class="divider"></div>

<section class="slide" id="slide5">
  <h2>Data quality & missingness (interactive)</h2>
  <p class="subtitle">
    Click a bar to pick a metric and instantly see which settlements have missing values.
    Switch “View” to explore highest/lowest values (e.g., high “No info”).
  </p>
  <div class="controls">
    <label>Metric
      <select id="quality_metric"></select>
    </label>
    <label>District
      <select id="quality_district"></select>
    </label>
    <label>View
      <select id="quality_view">
        <option value="missing">Missing values only</option>
        <option value="all">All settlements</option>
        <option value="highest">Highest values</option>
        <option value="lowest">Lowest values</option>
      </select>
    </label>
    <label>Search
      <input id="quality_search" type="text" placeholder="Type settlement name..." />
    </label>
    <label>Max rows
      <input id="quality_limit" type="range" min="10" max="203" step="1" value="60" />
      <span id="quality_limit_label">60</span>
    </label>
  </div>
  <p class="subtitle" id="quality_stats"></p>
  <div class="chart-container"><div id="chart_missing"></div></div>
  <div class="table-container">
    <div class="table-scroll" id="quality_table"></div>
  </div>
</section>
<div class="divider"></div>

<div class="nav">
  <a href="#slide1">Overview</a>
  <a href="#slide2">Ranking</a>
  <a href="#slide3">Relationships</a>
  <a href="#slide4">Distributions</a>
  <a href="#slide5">Data quality</a>
</div>

<script>
const EDUCATION_DATA = {data_json};
const METRICS = {metrics_json};
const SUMMARY = {summary_json};
const CORR = {corr_json};

const layoutDefaults = {{
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: {{ color: "#e6edf3", family: "IBM Plex Sans" }},
  margin: {{ l: 70, r: 30, t: 55, b: 60 }},
  hovermode: "closest",
  hoverdistance: 20
}};

function fmtNum(val) {{
  if (val === null || val === undefined || !isFinite(val)) return "";
  return (+val).toFixed(2);
}}

function isFiniteNumber(val) {{
  if (val === null || val === undefined) return false;
  const n = +val;
  return Number.isFinite(n);
}}

function metricSpec(key) {{
  return METRICS.find(m => m.key === key);
}}

function populateMetricSelect(selectEl, defaultKey) {{
  selectEl.innerHTML = "";
  for (const m of METRICS) {{
    const opt = document.createElement("option");
    opt.value = m.key;
    opt.textContent = m.label;
    selectEl.appendChild(opt);
  }}
  if (defaultKey) selectEl.value = defaultKey;
}}

function bestIsHigh(key) {{
  const m = metricSpec(key);
  return m ? !!m.higher_is_better : true;
}}

function getCleanRows(key) {{
  return EDUCATION_DATA
    .map(r => {{
      const v = r[key];
      return {{ ...r, __val: (v === null || v === undefined ? NaN : +v) }};
    }})
    .filter(r => isFinite(r.__val));
}}

function renderCorrelation() {{
  const keys = CORR.cols;
  const full = keys.map(c => (metricSpec(c)?.label) || c);
  const short = keys.map(c => (metricSpec(c)?.short) || (metricSpec(c)?.label) || c);
  const z = CORR.matrix;

  const annotations = [];
  for (let i = 0; i < full.length; i++) {{
    for (let j = 0; j < full.length; j++) {{
      const value = z[i][j];
      if (value === null || Number.isNaN(value)) continue;
      annotations.push({{
        x: full[j],
        y: full[i],
        text: value.toFixed(2),
        showarrow: false,
        font: {{ color: "#f8f9fb", size: 12 }}
      }});
    }}
  }}

  Plotly.newPlot("chart_corr", [{{
    type: "heatmap",
    z: z,
    x: full,
    y: full,
    zmin: -1,
    zmax: 1,
    colorscale: "RdBu",
    reversescale: true,
    hovertemplate: "%{{y}} vs %{{x}}<br>Spearman r=%{{z:.2f}}<extra></extra>"
  }}], {{
    ...layoutDefaults,
    title: "Spearman correlation (education metrics)",
    xaxis: {{
      tickangle: -30,
      tickfont: {{ size: 11 }},
      tickvals: full,
      ticktext: short,
      automargin: true
    }},
    yaxis: {{
      autorange: "reversed",
      tickfont: {{ size: 11 }},
      tickvals: full,
      ticktext: short,
      automargin: true
    }},
    margin: {{ l: 220, r: 40, t: 60, b: 150 }},
    annotations: annotations,
    showlegend: false,
    height: 560
  }}, {{displayModeBar: true, responsive: true}});
}}

function renderRanking() {{
  const key = document.getElementById("rank_metric").value;
  const mode = document.getElementById("rank_mode").value;
  const n = +document.getElementById("rank_n").value;
  document.getElementById("rank_n_label").textContent = String(n);

  const rows = getCleanRows(key);
  const higherBetter = bestIsHigh(key);
  const wantBest = mode === "best";

  // Decide sort direction:
  // - if higher is better and want best => descending
  // - if higher is better and want worst => ascending
  // - if lower is better and want best => ascending
  // - if lower is better and want worst => descending
  let asc;
  if (higherBetter) asc = !wantBest;
  else asc = wantBest;

  rows.sort((a,b) => asc ? (a.__val - b.__val) : (b.__val - a.__val));
  const picked = rows.slice(0, n);
  // For horizontal bars, reverse to show "best" on top
  picked.reverse();

  const y = picked.map(r => `${{r.settlement_name}} (${{r.settlement_symbol}})`);
  const x = picked.map(r => r.__val);

  const trace = {{
    type: "bar",
    orientation: "h",
    x: x,
    y: y,
    marker: {{ color: "rgba(255,183,3,0.85)" }},
    hovertemplate:
      "<b>%{{y}}</b><br>" +
      `${{metricSpec(key)?.label || key}}: %{{x:.2f}}` +
      "<br>District: %{{customdata[0]}}" +
      "<br>Status: %{{customdata[1]}}" +
      "<extra></extra>",
    customdata: picked.map(r => [r.district, r.municipal_status])
  }};

  const title = (wantBest ? "Best" : "Worst") + ` — ${{metricSpec(key)?.label || key}}`;
  const layout = {{
    ...layoutDefaults,
    title,
    height: Math.max(540, 22 * y.length + 180),
    xaxis: {{ title: "Percent", gridcolor: "rgba(255,255,255,0.08)", automargin: true }},
    yaxis: {{ title: "", automargin: true }},
    margin: {{ l: 320, r: 30, t: 70, b: 60 }},
  }};
  Plotly.newPlot("chart_rank", [trace], layout, {{displayModeBar: true, responsive: true}});
}}

function distinctDistricts() {{
  const s = new Set();
  for (const r of EDUCATION_DATA) {{
    if (r.district) s.add(r.district);
  }}
  return Array.from(s).sort();
}}

function renderScatter() {{
  const xKey = document.getElementById("scatter_x").value;
  const yKey = document.getElementById("scatter_y").value;
  const sizeKey = document.getElementById("scatter_size").value || null;

  function toNum(val) {{
    if (val === null || val === undefined) return NaN;
    const n = +val;
    return Number.isFinite(n) ? n : NaN;
  }}

  function toNumOrNull(val) {{
    const n = toNum(val);
    return Number.isFinite(n) ? n : null;
  }}

  const groups = new Map();
  let plotted = 0;
  let missingX = 0;
  let missingY = 0;
  let missingBoth = 0;
  for (const r of EDUCATION_DATA) {{
    const x = toNum(r[xKey]);
    const y = toNum(r[yKey]);
    const xOk = Number.isFinite(x);
    const yOk = Number.isFinite(y);
    if (!xOk) missingX += 1;
    if (!yOk) missingY += 1;
    if (!xOk && !yOk) missingBoth += 1;
    if (!xOk || !yOk) continue;
    plotted += 1;
    const d = r.district || "Unknown";
    if (!groups.has(d)) groups.set(d, []);
    groups.get(d).push({{ ...r, __x: x, __y: y }});
  }}

  const traces = [];
  for (const [district, rows] of groups.entries()) {{
    const x = rows.map(r => r.__x);
    const y = rows.map(r => r.__y);
    let size = undefined;
    if (sizeKey) {{
      size = rows.map(r => {{
        const vv = toNum(r[sizeKey]);
        const v = Number.isFinite(vv) ? vv : 0;
        return 6 + Math.min(28, v * 0.6);
      }});
    }}
    traces.push({{
      type: "scatter",
      mode: "markers",
      name: district,
      x,
      y,
      text: rows.map(r => `${{r.settlement_name}} (${{r.settlement_symbol}})`),
      marker: {{
        size: size || 10,
        opacity: 0.75,
        line: {{ width: 0.6, color: "rgba(255,255,255,0.25)" }},
      }},
      hovertemplate:
        "<b>%{{text}}</b><br>" +
        `${{metricSpec(xKey)?.label || xKey}}: %{{x:.2f}}` +
        "<br>" +
        `${{metricSpec(yKey)?.label || yKey}}: %{{y:.2f}}` +
        (sizeKey ? `<br>${{metricSpec(sizeKey)?.label || sizeKey}}: %{{customdata:.2f}}` : "") +
        "<extra></extra>",
      customdata: sizeKey ? rows.map(r => toNumOrNull(r[sizeKey])) : undefined,
    }});
  }}

  const layout = {{
    ...layoutDefaults,
    title: `${{metricSpec(xKey)?.label || xKey}} vs ${{metricSpec(yKey)?.label || yKey}}`,
    height: 620,
    xaxis: {{ title: "Percent", gridcolor: "rgba(255,255,255,0.08)", automargin: true }},
    yaxis: {{ title: "Percent", gridcolor: "rgba(255,255,255,0.08)", automargin: true }},
    legend: {{
      bgcolor: "rgba(8,12,18,0.6)",
      bordercolor: "rgba(255,255,255,0.18)",
      borderwidth: 1,
      x: 1.02,
      y: 1,
      xanchor: "left",
      yanchor: "top"
    }},
    margin: {{ l: 80, r: 280, t: 70, b: 70 }}
  }};
  Plotly.newPlot("chart_scatter", traces, layout, {{displayModeBar: true, responsive: true}});

  const total = EDUCATION_DATA.length;
  const statsEl = document.getElementById("scatter_stats");
  if (statsEl) {{
    statsEl.textContent =
      "Plotted: " + plotted + "/" + total +
      ". Missing X: " + missingX +
      ". Missing Y: " + missingY +
      ". Missing both: " + missingBoth + ".";
  }}
}}

function renderHistogram() {{
  const key = document.getElementById("hist_metric").value;
  const rows = getCleanRows(key);
  const x = rows.map(r => r.__val);
  const trace = {{
    type: "histogram",
    x,
    nbinsx: 28,
    marker: {{ color: "rgba(46,196,182,0.75)", line: {{ width: 0.5, color: "rgba(255,255,255,0.25)" }} }},
    hovertemplate: "Count=%{{y}}<br>Range start=%{{x}}<extra></extra>"
  }};
  const layout = {{
    ...layoutDefaults,
    title: `Distribution — ${{metricSpec(key)?.label || key}}`,
    height: 540,
    xaxis: {{ title: "Percent", gridcolor: "rgba(255,255,255,0.08)" }},
    yaxis: {{ title: "Settlements", gridcolor: "rgba(255,255,255,0.08)" }},
  }};
  Plotly.newPlot("chart_hist", [trace], layout, {{displayModeBar: true, responsive: true}});
}}

function setupNav() {{
  const navLinks = Array.from(document.querySelectorAll(".nav a"));
  const navLookup = new Map(navLinks.map(link => [link.getAttribute("href").slice(1), link]));

  function setActiveNav(id) {{
    navLinks.forEach(link => link.classList.remove("active"));
    const target = navLookup.get(id);
    if (target) target.classList.add("active");
  }}

  const observer = new IntersectionObserver((entries) => {{
    entries.forEach(entry => {{
      if (entry.isIntersecting) setActiveNav(entry.target.id);
    }});
  }}, {{ threshold: 0.45 }});

  document.querySelectorAll(".slide").forEach(slide => observer.observe(slide));
  if (location.hash) setActiveNav(location.hash.slice(1));
  else if (navLinks.length) navLinks[0].classList.add("active");
}}

function getRowsByDistrict(district) {{
  if (!district) return EDUCATION_DATA;
  return EDUCATION_DATA.filter(r => (r.district || "") === district);
}}

function renderMissingSummary(rows) {{
  const xLabels = METRICS.map(m => m.short || m.label || m.key);
  const fullLabels = METRICS.map(m => m.label || m.key);
  const keys = METRICS.map(m => m.key);
  const missingCounts = keys.map(key => rows.reduce((acc, r) => acc + (isFiniteNumber(r[key]) ? 0 : 1), 0));

  const trace = {{
    type: "bar",
    x: xLabels,
    y: missingCounts,
    marker: {{ color: "rgba(255,107,107,0.72)" }},
    customdata: keys,
    hovertemplate:
      "<b>%{{customdata}}</b><br>" +
      "%{{x}}<br>" +
      "%{{y}} settlements missing<br>" +
      "<extra></extra>"
  }};

  const maxY = Math.max(1, ...missingCounts);
  const layout = {{
    ...layoutDefaults,
    title: "Missing values by metric (click a bar to explore)",
    height: 520,
    margin: {{ l: 70, r: 30, t: 70, b: 140 }},
    xaxis: {{
      tickangle: -25,
      automargin: true,
      tickfont: {{ size: 11 }},
    }},
    yaxis: {{
      title: "Settlements (count)",
      gridcolor: "rgba(255,255,255,0.08)",
      rangemode: "tozero",
      range: [0, maxY * 1.15],
      automargin: true,
    }},
  }};

  const el = document.getElementById("chart_missing");
  Plotly.newPlot(el, [trace], layout, {{displayModeBar: true, responsive: true}});

  if (el && typeof el.removeAllListeners === "function") {{
    el.removeAllListeners("plotly_click");
  }}
  if (el && typeof el.on === "function") {{
    el.on("plotly_click", (evt) => {{
      const point = evt?.points?.[0];
      const key = point?.customdata;
      if (!key) return;
      const metricSelect = document.getElementById("quality_metric");
      const viewSelect = document.getElementById("quality_view");
      if (metricSelect) metricSelect.value = key;
      if (viewSelect) viewSelect.value = "missing";
      renderQuality();
      document.getElementById("quality_table")?.scrollTo({{ top: 0, behavior: "smooth" }});
    }});
  }}
}}

function renderQualityTable(rows) {{
  const metricKey = document.getElementById("quality_metric").value;
  const view = document.getElementById("quality_view").value;
  const q = (document.getElementById("quality_search").value || "").trim().toLowerCase();
  const limit = +document.getElementById("quality_limit").value;
  document.getElementById("quality_limit_label").textContent = String(limit);

  const nameMatches = (name) => !q || String(name || "").toLowerCase().includes(q);

  const base = rows.filter(r => nameMatches(r.settlement_name));
  const withVal = base.filter(r => isFiniteNumber(r[metricKey]));
  const missing = base.filter(r => !isFiniteNumber(r[metricKey]));

  let picked = [];
  if (view === "missing") {{
    picked = missing.slice();
  }} else if (view === "all") {{
    picked = base.slice();
  }} else if (view === "highest") {{
    picked = withVal.slice().sort((a,b) => {{
      const dv = (+b[metricKey]) - (+a[metricKey]);
      if (dv !== 0) return dv;
      return String(a.settlement_name || "").localeCompare(String(b.settlement_name || ""));
    }});
  }} else if (view === "lowest") {{
    picked = withVal.slice().sort((a,b) => {{
      const dv = (+a[metricKey]) - (+b[metricKey]);
      if (dv !== 0) return dv;
      return String(a.settlement_name || "").localeCompare(String(b.settlement_name || ""));
    }});
  }} else {{
    picked = base.slice();
  }}

  if (view === "missing" || view === "all") {{
    picked.sort((a,b) => {{
      const ad = (a.district || "").localeCompare(b.district || "");
      if (ad !== 0) return ad;
      return String(a.settlement_name || "").localeCompare(String(b.settlement_name || ""));
    }});
  }}

  const shown = picked.slice(0, limit);
  const metricLabel = metricSpec(metricKey)?.label || metricKey;

  const statsEl = document.getElementById("quality_stats");
  if (statsEl) {{
    statsEl.innerHTML =
      `<span class="pill ok">Filtered: <b>${{base.length}}</b></span> ` +
      `<span class="pill ok">With value: <b>${{withVal.length}}</b></span> ` +
      `<span class="pill missing">Missing: <b>${{missing.length}}</b></span> ` +
      `&nbsp;&nbsp; | &nbsp;&nbsp;` +
      `Metric: <b>${{metricLabel}}</b>`;
  }}

  const tableEl = document.getElementById("quality_table");
  if (!tableEl) return;

  const rowsHtml = shown.map(r => {{
    const val = isFiniteNumber(r[metricKey]) ? fmtNum(r[metricKey]) : "<span class='pill missing'>missing</span>";
    const noInfo = isFiniteNumber(r.edu_attain_pct_no_info) ? fmtNum(r.edu_attain_pct_no_info) : "";
    return `
      <tr>
        <td><strong>${{r.settlement_name}}</strong></td>
        <td>${{r.settlement_symbol}}</td>
        <td>${{r.district}}</td>
        <td>${{r.municipal_status}}</td>
        <td>${{val}}</td>
        <td>${{noInfo}}</td>
      </tr>
    `;
  }}).join("");

  tableEl.innerHTML = `
    <table>
      <thead>
        <tr>
          <th style="width: 30%;">Settlement</th>
          <th style="width: 12%;">Symbol</th>
          <th style="width: 18%;">District</th>
          <th style="width: 20%;">Status</th>
          <th style="width: 12%;">Value</th>
          <th style="width: 8%;">No info</th>
        </tr>
      </thead>
      <tbody>
        ${{rowsHtml || `<tr><td colspan="6">No rows match the current filters.</td></tr>`}}
      </tbody>
    </table>
  `;
}}

function renderQuality() {{
  const district = document.getElementById("quality_district").value || "";
  const rows = getRowsByDistrict(district);
  renderMissingSummary(rows);
  renderQualityTable(rows);
}}

function setup() {{
  const summaryEl = document.getElementById("summary_box");
  summaryEl.innerHTML =
    `<b>Settlements:</b> ${{SUMMARY.n_settlements}}<br>` +
    `<b>Districts:</b> ${{SUMMARY.n_districts}}<br>` +
    `<b>Input:</b> <code>${{SUMMARY.input_file}}</code>`;

  // selects
  populateMetricSelect(document.getElementById("rank_metric"), "edu_attain_pct_academic_degree");
  populateMetricSelect(document.getElementById("scatter_x"), "edu_bagrut_uni_req_pct");
  populateMetricSelect(document.getElementById("scatter_y"), "edu_attain_pct_academic_degree");
  populateMetricSelect(document.getElementById("hist_metric"), "edu_attain_pct_bagrut_or_higher");
  populateMetricSelect(document.getElementById("quality_metric"), "edu_attain_pct_no_info");

  // district select
  const districtSelect = document.getElementById("quality_district");
  districtSelect.innerHTML = "";
  const allOpt = document.createElement("option");
  allOpt.value = "";
  allOpt.textContent = "All districts";
  districtSelect.appendChild(allOpt);
  for (const d of distinctDistricts()) {{
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d;
    districtSelect.appendChild(opt);
  }}

  document.getElementById("rank_metric").addEventListener("change", renderRanking);
  document.getElementById("rank_mode").addEventListener("change", renderRanking);
  document.getElementById("rank_n").addEventListener("input", renderRanking);

  document.getElementById("scatter_x").addEventListener("change", renderScatter);
  document.getElementById("scatter_y").addEventListener("change", renderScatter);
  document.getElementById("scatter_size").addEventListener("change", renderScatter);

  document.getElementById("hist_metric").addEventListener("change", renderHistogram);

  document.getElementById("quality_metric").addEventListener("change", renderQuality);
  document.getElementById("quality_district").addEventListener("change", renderQuality);
  document.getElementById("quality_view").addEventListener("change", renderQuality);
  document.getElementById("quality_search").addEventListener("input", renderQuality);
  document.getElementById("quality_limit").addEventListener("input", renderQuality);

  renderCorrelation();
  renderRanking();
  renderScatter();
  renderHistogram();
  renderQuality();
  setupNav();
}}

setup();
</script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote: {output_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
