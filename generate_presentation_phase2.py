"""
Generate Phase 2 HTML presentation: Education, Income, and Demographics.
Reads benefits_final.csv (built by build_master_dataset.py) and produces
presentation_phase2.html with 6 interactive slides.
"""

from __future__ import annotations

import html as html_lib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = (
    PROJECT_ROOT
    / "datas_for_research_vicious_circle_project"
    / "data"
    / "processed"
    / "benefits_final.csv"
)


# ── helpers ──────────────────────────────────────────────────────────────────

def linear_fit(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x_vals = np.asarray(x, dtype=float)
    y_vals = np.asarray(y, dtype=float)
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals, y_vals = x_vals[mask], y_vals[mask]
    if len(x_vals) == 0:
        return 0.0, 0.0
    a = np.vstack([x_vals, np.ones(len(x_vals))]).T
    slope, intercept = np.linalg.lstsq(a, y_vals, rcond=None)[0]
    return float(slope), float(intercept)


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="pearson"))


def fmt_num(val: float, digits: int = 2) -> str:
    if pd.isna(val):
        return ""
    return f"{float(val):.{digits}f}"


def fmt_int(val: float) -> str:
    if pd.isna(val):
        return ""
    return f"{int(round(float(val))):,}"


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


# ── load data ────────────────────────────────────────────────────────────────

print("Loading data from benefits_final.csv ...")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df)} rows, {len(df.columns)} columns")

# Ensure numeric types
numeric_cols = [
    "total_population", "population_0_17", "population_18_64", "population_65_plus",
    "general_disability_benefit", "special_services_for_persons_with_disabilities",
    "mobility_benefit", "income_support_benefit", "long_term_care_benefit",
    "socio_economic_index_score", "socio_economic_index_cluster",
    "peripherality_index_score", "peripherality_index_cluster",
    "edu_attain_pct_academic_degree", "average_monthly_salary_2023",
    "arab_population_percentage", "haredi_population_percentage",
    "jewish_population_percentage", "jewish_non_haredi_population_percentage",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Calculate rates
df["general_disability_rate"] = (
    df["general_disability_benefit"] / df["population_18_64"] * 100
).round(2)
df["income_support_rate"] = (
    df["income_support_benefit"] / df["population_18_64"] * 100
).round(2)

# ── Slide 1: Data Overview ──────────────────────────────────────────────────

n_total = len(df)
n_edu = int(df["edu_attain_pct_academic_degree"].notna().sum())
n_salary = int(df["average_monthly_salary_2023"].notna().sum())
n_arab = int(df["arab_population_percentage"].notna().sum())
n_haredi = int(df["haredi_population_percentage"].notna().sum())
n_disability = int(df["general_disability_rate"].notna().sum())
n_se = int(df["socio_economic_index_score"].notna().sum())

overview_rows = [
    ["Academic Degree %", str(n_edu), f"{n_edu}/{n_total}"],
    ["Average Monthly Salary", str(n_salary), f"{n_salary}/{n_total}"],
    ["Arab Population %", str(n_arab), f"{n_arab}/{n_total}"],
    ["Haredi Population %", str(n_haredi), f"{n_haredi}/{n_total}"],
    ["General Disability Rate", str(n_disability), f"{n_disability}/{n_total}"],
    ["Socio-Economic Score", str(n_se), f"{n_se}/{n_total}"],
]
overview_table = render_table(["Variable", "Available (n)", "Coverage"], overview_rows)

# ── Slide 2: Education & Disability ─────────────────────────────────────────

print("Preparing Slide 2: Education & Disability ...")
df_edu = df[
    ["edu_attain_pct_academic_degree", "general_disability_rate",
     "socio_economic_index_cluster", "settlement_name"]
].copy()
for col in ["edu_attain_pct_academic_degree", "general_disability_rate", "socio_economic_index_cluster"]:
    df_edu[col] = pd.to_numeric(df_edu[col], errors="coerce")
df_edu = df_edu.dropna(subset=["edu_attain_pct_academic_degree", "general_disability_rate", "socio_economic_index_cluster"])

edu_slope, edu_intercept = linear_fit(
    df_edu["edu_attain_pct_academic_degree"], df_edu["general_disability_rate"]
)
edu_corr = spearman_corr(
    df_edu["edu_attain_pct_academic_degree"], df_edu["general_disability_rate"]
)
edu_data = {
    "x": df_edu["edu_attain_pct_academic_degree"].round(2).tolist(),
    "y": df_edu["general_disability_rate"].round(2).tolist(),
    "names": df_edu["settlement_name"].tolist(),
    "cluster": df_edu["socio_economic_index_cluster"].astype(float).tolist(),
    "slope": edu_slope,
    "intercept": edu_intercept,
    "corr": edu_corr,
}

# ── Slide 3: Salary & Disability ────────────────────────────────────────────

print("Preparing Slide 3: Salary & Disability ...")
df_sal = df[
    ["average_monthly_salary_2023", "general_disability_rate",
     "socio_economic_index_cluster", "settlement_name"]
].copy()
for col in ["average_monthly_salary_2023", "general_disability_rate", "socio_economic_index_cluster"]:
    df_sal[col] = pd.to_numeric(df_sal[col], errors="coerce")
df_sal = df_sal.dropna(subset=["average_monthly_salary_2023", "general_disability_rate", "socio_economic_index_cluster"])

sal_slope, sal_intercept = linear_fit(
    df_sal["average_monthly_salary_2023"], df_sal["general_disability_rate"]
)
sal_corr = spearman_corr(
    df_sal["average_monthly_salary_2023"], df_sal["general_disability_rate"]
)
# Also compute salary vs SE score correlation for the insight box
df_sal_se = df[["average_monthly_salary_2023", "socio_economic_index_score"]].dropna()
sal_se_corr = spearman_corr(
    df_sal_se["average_monthly_salary_2023"], df_sal_se["socio_economic_index_score"]
)
sal_data = {
    "x": df_sal["average_monthly_salary_2023"].round(0).tolist(),
    "y": df_sal["general_disability_rate"].round(2).tolist(),
    "names": df_sal["settlement_name"].tolist(),
    "cluster": df_sal["socio_economic_index_cluster"].astype(float).tolist(),
    "slope": sal_slope,
    "intercept": sal_intercept,
    "corr": sal_corr,
    "sal_se_corr": sal_se_corr,
}

# ── Slide 4: Demographic Divide (Arab %) ────────────────────────────────────

print("Preparing Slide 4: Demographic Divide ...")
df_arab = df[
    ["arab_population_percentage", "general_disability_rate",
     "socio_economic_index_cluster", "settlement_name"]
].copy()
for col in ["arab_population_percentage", "general_disability_rate", "socio_economic_index_cluster"]:
    df_arab[col] = pd.to_numeric(df_arab[col], errors="coerce")
df_arab = df_arab.dropna(subset=["arab_population_percentage", "general_disability_rate", "socio_economic_index_cluster"])

arab_slope, arab_intercept = linear_fit(
    df_arab["arab_population_percentage"], df_arab["general_disability_rate"]
)
arab_corr = spearman_corr(
    df_arab["arab_population_percentage"], df_arab["general_disability_rate"]
)
arab_data = {
    "x": df_arab["arab_population_percentage"].round(2).tolist(),
    "y": df_arab["general_disability_rate"].round(2).tolist(),
    "names": df_arab["settlement_name"].tolist(),
    "cluster": df_arab["socio_economic_index_cluster"].astype(float).tolist(),
    "slope": arab_slope,
    "intercept": arab_intercept,
    "corr": arab_corr,
}

# ── Slide 5: The Haredi Puzzle ───────────────────────────────────────────────

print("Preparing Slide 5: The Haredi Puzzle ...")
df_haredi = df[
    ["haredi_population_percentage", "general_disability_rate",
     "socio_economic_index_cluster", "settlement_name", "socio_economic_index_score"]
].copy()
for col in ["haredi_population_percentage", "general_disability_rate",
            "socio_economic_index_cluster", "socio_economic_index_score"]:
    df_haredi[col] = pd.to_numeric(df_haredi[col], errors="coerce")
df_haredi = df_haredi.dropna(subset=[
    "haredi_population_percentage", "general_disability_rate", "socio_economic_index_cluster"
])

haredi_slope, haredi_intercept = linear_fit(
    df_haredi["haredi_population_percentage"], df_haredi["general_disability_rate"]
)
haredi_corr = spearman_corr(
    df_haredi["haredi_population_percentage"], df_haredi["general_disability_rate"]
)
haredi_data = {
    "x": df_haredi["haredi_population_percentage"].round(2).tolist(),
    "y": df_haredi["general_disability_rate"].round(2).tolist(),
    "names": df_haredi["settlement_name"].tolist(),
    "cluster": df_haredi["socio_economic_index_cluster"].astype(float).tolist(),
    "slope": haredi_slope,
    "intercept": haredi_intercept,
    "corr": haredi_corr,
}

# ── Slide 6: Correlation Matrix ─────────────────────────────────────────────

print("Preparing Slide 6: Correlation Matrix ...")
corr_vars = {
    "edu_attain_pct_academic_degree": "Academic Degree %",
    "average_monthly_salary_2023": "Avg Monthly Salary",
    "arab_population_percentage": "Arab Pop %",
    "haredi_population_percentage": "Haredi Pop %",
    "jewish_non_haredi_population_percentage": "Jewish Non-Haredi %",
    "general_disability_rate": "General Disability Rate",
    "income_support_rate": "Income Support Rate",
    "socio_economic_index_score": "SE Index Score",
    "peripherality_index_score": "Peripherality Score",
}

# Filter to columns that actually exist
corr_cols = [c for c in corr_vars if c in df.columns]
corr_labels = [corr_vars[c] for c in corr_cols]

corr_matrix = df[corr_cols].corr(method="spearman")
heatmap_data = {
    "z": corr_matrix.values.round(3).tolist(),
    "x": corr_labels,
    "y": corr_labels,
}


# ── HTML generation ──────────────────────────────────────────────────────────

print("Generating HTML ...")

slides: list[str] = []
nav_items: list[tuple[str, str]] = []


def add_slide(slide_id: str, nav_label: str, html_block: str) -> None:
    slides.append(html_block)
    nav_items.append((nav_label, slide_id))


# Slide 1: Title + Overview
slide1 = f"""
<section class="slide" id="slide1">
  <div class="title-block">
    <h1>Phase 2: Beyond the Vicious Circle</h1>
    <p class="subtitle">Education, Income, and Demographics across {n_total} Israeli Settlements</p>
    <p class="meta">Data: Bituach Leumi (Dec 2024) + CBS + LAMAS</p>
    <div class="intro-lead">
      <p>Phase 1 established the correlation between socio-economic vulnerability and disability
      (Spearman r = 0.58). In Phase 2 we ask: <strong>what drives the vicious circle?</strong>
      We add education levels, average salaries, and demographic composition (Arab %, Haredi %)
      to understand which factors are independent predictors and which are proxies for the same
      underlying signal.</p>
    </div>
  </div>
  <div class="panel" style="max-width:700px;">
    <h3>New Variables &amp; Coverage</h3>
    {overview_table}
  </div>
</section>
<div class="divider"></div>
"""
add_slide("slide1", "Intro", slide1)

# Slide 2: Education & Disability
slide2 = f"""
<section class="slide" id="slide2">
  <h2>Education &amp; Disability</h2>
  <p class="subtitle">% with Academic Degree vs General Disability Rate | Spearman r = {edu_data["corr"]:.2f}</p>
  <div class="insight">
    <h3>Key Finding</h3>
    <p>Higher education correlates with lower disability rates (r = {edu_data["corr"]:.2f}).
    But is education a <em>cause</em> of better health, or a <em>consequence</em> of better
    socio-economic conditions? Settlements with more graduates tend to be wealthier — so education
    may be a proxy for the same structural advantage captured by the SE index.</p>
  </div>
  <div class="chart-container"><div id="graph_edu"></div></div>
</section>
<div class="divider"></div>
"""
add_slide("slide2", "Education", slide2)

# Slide 3: Salary & Disability
slide3 = f"""
<section class="slide" id="slide3">
  <h2>Salary &amp; Disability</h2>
  <p class="subtitle">Average Monthly Salary vs General Disability Rate | Spearman r = {sal_data["corr"]:.2f}</p>
  <div class="insight">
    <h3>Key Finding</h3>
    <p>Salary is almost perfectly correlated with the SE index (r = {sal_se_corr:.2f}) — it is
    essentially the <strong>same signal</strong>. This confirms that salary does not add independent
    explanatory power: it is a component of socio-economic status, not a separate driver.</p>
  </div>
  <div class="chart-container"><div id="graph_salary"></div></div>
</section>
<div class="divider"></div>
"""
add_slide("slide3", "Salary", slide3)

# Slide 4: Demographic Divide
slide4 = f"""
<section class="slide" id="slide4">
  <h2>The Demographic Divide</h2>
  <p class="subtitle">Arab Population % vs General Disability Rate | Spearman r = {arab_data["corr"]:.2f}</p>
  <div class="insight">
    <h3>Key Finding</h3>
    <p>Arab-majority settlements show higher disability rates — but is this ethnicity or poverty?
    Arab settlements are disproportionately represented in low SE clusters.
    The correlation (r = {arab_data["corr"]:.2f}) may largely reflect the overlap between
    Arab population concentration and structural disadvantage.</p>
  </div>
  <div class="chart-container"><div id="graph_arab"></div></div>
</section>
<div class="divider"></div>
"""
add_slide("slide4", "Demographics", slide4)

# Slide 5: Haredi Puzzle
slide5 = f"""
<section class="slide" id="slide5">
  <h2>The Haredi Puzzle</h2>
  <p class="subtitle">Haredi Population % vs General Disability Rate | Spearman r = {haredi_data["corr"]:.2f}</p>
  <div class="insight">
    <h3>Key Finding</h3>
    <p>Haredi communities have <strong>low</strong> socio-economic scores but only
    <strong>slightly</strong> elevated disability rates. They break the vicious circle pattern.
    The weak correlation (r = {haredi_data["corr"]:.2f}) suggests that community-level factors
    — social cohesion, family support networks, or different patterns of benefit claiming —
    may buffer against the poverty-disability link observed in other populations.</p>
  </div>
  <div class="chart-container"><div id="graph_haredi"></div></div>
</section>
<div class="divider"></div>
"""
add_slide("slide5", "Haredi", slide5)

# Slide 6: Correlation Matrix
slide6 = """
<section class="slide" id="slide6">
  <h2>Correlation Matrix: All Variables</h2>
  <p class="subtitle">Spearman rank correlations — which variables are redundant?</p>
  <div class="insight">
    <h3>Reading the Matrix</h3>
    <p>Dark blue = strong positive correlation, dark red = strong negative.
    Notice how Salary and SE Score are nearly identical (r ~ 0.94), while
    Haredi % behaves independently from most other variables.</p>
  </div>
  <div class="chart-container"><div id="graph_heatmap"></div></div>
</section>
<div class="divider"></div>
"""
add_slide("slide6", "Correlations", slide6)

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
  <title>Phase 2: Beyond the Vicious Circle</title>
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
    .insight, .warning, .panel {{
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

# ── JavaScript / Plotly ──────────────────────────────────────────────────────

js_blocks: list[str] = []

js_blocks.append(f"""
<script>
const eduData = {json.dumps(edu_data)};
const salData = {json.dumps(sal_data)};
const arabData = {json.dumps(arab_data)};
const harediData = {json.dumps(haredi_data)};
const heatmapData = {json.dumps(heatmap_data)};

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

function range(arr) {{
  let min = Infinity;
  let max = -Infinity;
  arr.forEach(v => {{
    if (v < min) min = v;
    if (v > max) max = v;
  }});
  return [min, max];
}}

function linePoints(xArr, slope, intercept) {{
  const [minX, maxX] = range(xArr);
  const xLine = [minX, maxX];
  const yLine = xLine.map(x => slope * x + intercept);
  return {{ xLine, yLine }};
}}

function scatterWithTrend(divId, data, xTitle, yTitle, chartTitle, corrLabel) {{
  const line = linePoints(data.x, data.slope, data.intercept);
  Plotly.newPlot(divId, [
    {{
      x: data.x,
      y: data.y,
      mode: "markers",
      type: "scatter",
      text: data.names,
      cliponaxis: true,
      marker: {{
        color: data.cluster,
        colorscale: "Viridis",
        size: 9,
        showscale: true,
        colorbar: {{
          ...colorbarOutside,
          title: {{ text: "SE Cluster" }},
          len: 0.85
        }},
        opacity: 0.85,
        line: {{ color: "rgba(255,255,255,0.5)", width: 0.7 }}
      }},
      hovertemplate: "<b>%{{text}}</b><br>" + xTitle + ": %{{x}}<br>" + yTitle + ": %{{y:.2f}}%<extra></extra>"
    }},
    {{
      x: line.xLine,
      y: line.yLine,
      mode: "lines",
      type: "scatter",
      line: {{ color: "#ff6b6b", width: 3 }},
      hoverinfo: "skip",
      showlegend: false
    }}
  ], {{
    ...layoutDefaults,
    title: chartTitle,
    xaxis: {{
      title: {{ text: xTitle, standoff: 10 }},
      gridcolor: "rgba(255,255,255,0.08)",
      automargin: true,
      domain: [0, 0.86]
    }},
    yaxis: {{
      title: {{ text: yTitle, standoff: 10 }},
      gridcolor: "rgba(255,255,255,0.08)",
      automargin: true
    }},
    annotations: [{{
      text: corrLabel,
      xref: "paper",
      yref: "paper",
      x: 0.02,
      y: 0.98,
      xanchor: "left",
      yanchor: "top",
      showarrow: false,
      font: {{ size: 14, color: "#ffb703" }},
      bgcolor: "rgba(0,0,0,0.5)",
      borderpad: 6,
      bordercolor: "rgba(255,183,3,0.4)",
      borderwidth: 1
    }}],
    margin: {{ l: 70, r: 130, t: 60, b: 70 }},
    showlegend: false,
    height: 560
  }}, {{ responsive: true }});
}}
</script>
""")

# Scatter plots
js_blocks.append("""
<script>
scatterWithTrend(
  "graph_edu", eduData,
  "% with Academic Degree", "General Disability Rate (%)",
  "Education vs Disability",
  "Spearman r = " + eduData.corr.toFixed(2)
);

scatterWithTrend(
  "graph_salary", salData,
  "Average Monthly Salary (NIS)", "General Disability Rate (%)",
  "Salary vs Disability",
  "Spearman r = " + salData.corr.toFixed(2)
);

scatterWithTrend(
  "graph_arab", arabData,
  "Arab Population %", "General Disability Rate (%)",
  "Arab Population % vs Disability",
  "Spearman r = " + arabData.corr.toFixed(2)
);

scatterWithTrend(
  "graph_haredi", harediData,
  "Haredi Population %", "General Disability Rate (%)",
  "Haredi Population % vs Disability",
  "Spearman r = " + harediData.corr.toFixed(2)
);
</script>
""")

# Heatmap
js_blocks.append("""
<script>
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
      font: {
        size: 10,
        color: Math.abs(value) > 0.5 ? "#fff" : "#ccc"
      }
    });
  }
}

Plotly.newPlot("graph_heatmap", [
  {
    z: heatmapData.z,
    x: heatmapData.x,
    y: heatmapData.y,
    type: "heatmap",
    colorscale: [
      [0.0, "#313695"],
      [0.25, "#4575b4"],
      [0.5, "#f7f7f7"],
      [0.75, "#d73027"],
      [1.0, "#a50026"]
    ],
    zmin: -1,
    zmax: 1,
    hoverongaps: false,
    hovertemplate: "%{x}<br>%{y}<br>r = %{z:.2f}<extra></extra>",
    colorbar: {
      ...colorbarOutside,
      title: { text: "Spearman r" }
    }
  }
], {
  ...layoutDefaults,
  title: "Spearman Correlation Matrix",
  xaxis: {
    tickangle: -40,
    automargin: true,
    domain: [0, 0.86]
  },
  yaxis: {
    automargin: true
  },
  annotations: heatmapAnnotations,
  margin: { l: 160, r: 130, t: 60, b: 140 },
  height: 620
}, { responsive: true });
</script>
""")

# Navigation scroll-spy
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

# ── Write output ─────────────────────────────────────────────────────────────

js_content = "\n".join(js_blocks)
html_content = f"{html_head}{slides_html}{nav_html}{js_content}{html_tail}"

output_path = PROJECT_ROOT / "presentation_phase2.html"
output_path.write_text(html_content, encoding="utf-8")
print(f"Presentation saved to: {output_path}")
