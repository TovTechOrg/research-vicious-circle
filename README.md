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
- `build_master_dataset.py` — builds the master dataset table
- `requirements.txt` — Python dependencies for scripts

## Google Colab (quick run)

Upload into the Colab workspace (`/content`):
- `build_master_dataset.py`
- `datas_for_research_vicious_circle_project/` (the raw data folder with the `.xlsx/.csv` inputs inside)

Then paste into **one cell** and run:

```python
!pip -q install pandas numpy openpyxl plotly

from pathlib import Path
from build_master_dataset import build_master_dataset

# Path to the folder that contains the raw files (xlsx/csv)
DATA_DIR = Path("datas_for_research_vicious_circle_project").resolve()
print("DATA_DIR =", DATA_DIR)

# Build (and save outputs into DATA_DIR / data / processed)
data_master = build_master_dataset(data_dir=DATA_DIR, save=True, verbose=True)
data_master.head()
```

Outputs will be created here:
- `datas_for_research_vicious_circle_project/data/processed/benefits_final.csv`
- `datas_for_research_vicious_circle_project/data/processed/benefits_final.xlsx` (only if you run with `--excel`)

Tip: if you want an Excel file too (easy to view/download from Colab):

```python
data_master = build_master_dataset(data_dir=DATA_DIR, save=True, verbose=True, write_excel=True, write_pkl=False)
```

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
