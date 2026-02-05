# Vicious Circle Research

**Research Question:** Is there a relationship between socio-economic vulnerability and disability rates in Israel? If so, where is it strongest and for which types of disability?

This project explores correlations between poverty indicators and health/disability outcomes across **278 Israeli settlements**, using administrative data from the National Insurance Institute (Bituach Leumi) and official indices from the Central Bureau of Statistics (CBS).

## Live Presentation

**[View interactive presentation](https://tovtechorg.github.io/research-vicious-circle/)**

> The website is served via GitHub Pages. `index.html` redirects to `presentation_main.html`.

## What We Found (Phase 1)

### 1) Strong association (Social ↔ Health)
- **Spearman r = 0.58** between composite Social Index and Health Index
- The relationship is **monotonic** — more distressed social conditions tend to align with worse health/disability outcomes
- Consistent across multiple disability indicators (general, mobility, special services)

### 2) Intergenerational pattern
- Localities with high **adult disability** (ages 18–64) often show high **child disability** (ages 0–17)
- Suggests potential intergenerational transmission of disadvantage
- A **Red Zone** is defined where both rates exceed national weighted averages

### 3) Non-linear relationship with wealth
- Disability rates are similar across Q1–Q3 (poorest 75% of settlements)
- Sharp drop only in Q4 (wealthiest quartile)
- Suggests a **threshold effect** — benefits of wealth appear only at higher levels

### 4) Outlier analysis
| Type | Examples | Interpretation |
|------|----------|----------------|
| **Service magnets** | Tiberias, Be'er Sheva | Regional hubs may carry a “regional burden” by attracting vulnerable populations seeking services and public housing |
| **Resilient communities** | Brukhin, Talmon, Ofra | Strong community networks and younger demographics may buffer against welfare dependency |

## What We Did NOT Prove

- **Causation** (correlation does not prove cause)
- A proven **self-reinforcing “vicious circle” mechanism**
- The **direction** of effect (poverty → disability vs disability → poverty)

> Longitudinal research would be needed to establish causal relationships.

## Methodology (Core Choices)

### Unit of analysis
- **Locality level** (not individual) — useful for spatial/policy patterns

### Why rates (not absolute counts)
- Rates normalize for locality size: **Rate = Recipients / Population × 100**

### Composite indices

Both indices are normalized to **[-1, +1]** (Distress → Resilience).

**Social Index** (higher = more resilient / stronger socio-economic position):
- 50% — Socio-Economic Score (CBS)
- 25% — Peripherality Index
- 25% — Income Support Rate (inverted)

**Health Index** (higher = healthier / lower disability burden):
- 50% — General Disability Rate (inverted)
- 25% — Special Services Disability Rate (inverted)
- 25% — Mobility Disability Rate (inverted)

## Data Sources

| Source | Data | Period |
|--------|------|--------|
| **Bituach Leumi** | Benefit recipients by settlement (>2,000 residents) | December 2024 |
| **CBS (LAMAS)** | Socio-Economic Index (cluster & score) | 2021 |
| **CBS (LAMAS)** | Peripherality Index (cluster & score) | 2020 |

## Repository Layout

Tracked (in GitHub):
- `presentation_main.html` — generated interactive presentation (served by Pages)
- `index.html` — redirect entrypoint
- `generate_presentation_insights.py` — generator script (builds `presentation_main.html`)
- `research_vicious_circle.ipynb` — main analysis notebook

## Setup (for running scripts)

```bash
python -m venv .venv
```

Activate:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Google Colab (quick run)

If you want to run the pipeline in Google Colab, upload:
- `build_master_dataset.py`
- `requirements.txt` (optional)
- `datas_for_research_vicious_circle_project/` (the raw data folder)

Then run in a single cell:

```python
!pip -q install -r requirements.txt
!python build_master_dataset.py
```

Local-only (gitignored):
- `datas_for_research_vicious_circle_project/` — raw input data files
- `local/` — scratch/legacy/diagnostics and temporary artifacts
- `.vscode/` — editor settings

## How to Update the Presentation

```bash
python generate_presentation_insights.py
```

This regenerates `presentation_main.html`. Push to GitHub and the live site updates automatically.

## How to Extract Education CSV (Standalone)

Extract education indicators from `p_libud_23.xlsx` into a standalone CSV (no merge with benefits / main dataset):

```bash
python extract_education_from_p_libud_23.py
```

Output (recommended columns): `datas_for_research_vicious_circle_project/education_from_p_libud_23.csv`

To export the full set of education attainment columns:

```bash
python extract_education_from_p_libud_23.py --all-columns
```

Output: `datas_for_research_vicious_circle_project/education_from_p_libud_23_full.csv`

Google Sheets locale tip: if your Sheets locale does not auto-split comma-separated CSVs, use:

```bash
python extract_education_from_p_libud_23.py --google-sheets
```

Output: `datas_for_research_vicious_circle_project/education_from_p_libud_23_google_sheets.csv`

If you prefer uploading an Excel file to Google Sheets, use:

```bash
python extract_education_from_p_libud_23.py --excel
```

Output: `datas_for_research_vicious_circle_project/education_from_p_libud_23_google_sheets.xlsx`

## How to Build the Master Dataset (benefits_final)

Build a single settlement-level table by combining Bituach Leumi benefits with CBS indices, coordinates, demographics and
education:

```bash
python build_master_dataset.py
```

To also save an Excel file (easy to inspect/share):

```bash
python build_master_dataset.py --excel
```

If you don't need the `.pkl` output:

```bash
python build_master_dataset.py --no-pkl
```

If your raw files live in a different folder (e.g., Google Drive / another machine), point the script to it:

```bash
python build_master_dataset.py --data-dir /path/to/datas_for_research_vicious_circle_project
```

You can also override individual inputs (example: different education file):

```bash
python build_master_dataset.py --education /path/to/education_from_p_libud_23.csv
```

Other override flags exist too (e.g. `--benefits`, `--lamas`, `--coordinates`, etc.) — run:

```bash
python build_master_dataset.py -h
```

Outputs (gitignored, local-only):
- `datas_for_research_vicious_circle_project/data/processed/benefits_final.csv`
- `datas_for_research_vicious_circle_project/data/processed/benefits_final.pkl`

From a notebook:

```python
from build_master_dataset import build_master_dataset

data_master = build_master_dataset(save=True)
```

Notebook tip: for non-standard paths, use:

```python
data_master = build_master_dataset(
    data_dir="/path/to/datas_for_research_vicious_circle_project",
    paths={"education": "/path/to/education_from_p_libud_23.csv"},
    save=True,
)
```

## Education Presentation

Generate an education-only interactive presentation:

```bash
python generate_education_presentation.py
```

Output: `presentation_education.html`

## Phase 2: Additional Data Collection

We are collecting additional data to deepen the analysis.

**Data Sources:**
- [Working Spreadsheet](https://docs.google.com/spreadsheets/d/19G-GOPbrKtfNGO_dl_m944ULK2tldN9S/edit?usp=sharing)
- [Haredi Population Statistics (IDI Yearbook 2023)](https://www.idi.org.il/haredi/2023/?chapter=51973#excel) — demographics, fertility rates, geographic distribution

## Links

- [Live Presentation](https://tovtechorg.github.io/research-vicious-circle/)
- [Project Kanban Board](https://docs.google.com/spreadsheets/d/1v2XxbDu7Gwhqv4Y4Dbebmn-HpkJRhlU1/edit?usp=sharing)
- [Phase 2 Working Spreadsheet](https://docs.google.com/spreadsheets/d/19G-GOPbrKtfNGO_dl_m944ULK2tldN9S/edit?usp=sharing)
- [Haredi Population Statistics (IDI 2023)](https://www.idi.org.il/haredi/2023/?chapter=51973#excel)
