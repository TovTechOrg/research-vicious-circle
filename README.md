# Vicious Circle Research

**Research Question:** Is there a relationship between socio-economic vulnerability and disability rates in Israel? If so, where is it strongest and for which types of disability?

This project explores correlations between poverty indicators and health/disability outcomes across **278 Israeli settlements**, using administrative data from the National Insurance Institute (Bituach Leumi) and the Central Bureau of Statistics.

---

## Live Presentation

**[View Interactive Presentation](https://tovtechorg.github.io/research-vicious-circle/)**

---

## What We Found

### 1. Strong Association (Social ↔ Health)
- **Spearman r = 0.58** between composite Social Index and Health Index
- The relationship is **monotonic** — distressed localities (lower social scores) tend to align with worse health outcomes (lower health scores)
- This is consistent across multiple disability indicators used inside the composite Health Index (general, mobility, special services)

### 2. Intergenerational Pattern
- Localities with high **adult disability** (ages 18-64) often show high **child disability** (ages 0-17)
- This suggests potential intergenerational transmission of disadvantage
- "Red Zone" localities identified where both rates exceed national averages

### 3. Non-Linear Relationship with Wealth
- Disability rates are similar across Q1-Q3 (poorest 75% of settlements)
- Sharp drop only in Q4 (wealthiest quartile)
- Suggests a **threshold effect** — benefits of wealth appear only at higher levels

### 4. Outlier Analysis
| Type | Examples | Explanation |
|------|----------|-------------|
| **"Service Magnets"** | Tiberias, Be'er Sheva | Regional hubs attract vulnerable populations seeking services; carry "regional burden" |
| **"Resilient Communities"** | Brukhin, Talmon, Ofra | Strong community networks and younger demographics prevent welfare dependency |

---

## What We Did NOT Prove

- **Causation** — correlation does not prove that poverty *causes* disability or vice versa
- **Cyclical mechanism** — we did not prove the "vicious circle" exists as a self-reinforcing loop
- **Direction of effect** — cannot determine if poverty leads to disability or disability leads to poverty

> Further **longitudinal research** would be needed to establish causal relationships.

---

## Methodology

### Unit of Analysis
- **Locality level** (not individual) — disability is a spatial, structural phenomenon influenced by socio-economic context
- Allows identification of regional patterns and policy-relevant insights

### Why Rates, Not Absolute Numbers?
- Absolute numbers are biased by locality size
- **Rate = Recipients / Population × 100** enables fair comparison between small and large settlements
- Question: "Where is the **risk** of disability higher?" not "Where are there more disabled people?"

### Types of Disability Analyzed
| Category | Included | Rationale |
|----------|----------|-----------|
| **Poverty-related** | General disability, special services, mobility, income support | Working-age disability with economic dependence |
| **Excluded** | Work injuries | Occupational events, not cumulative poverty process |
| **Separate analysis** | Long-term care (65+) | Age-dependent, analyzed separately from working-age |

### Composite Indices

Both indices are normalized to **[-1, +1]** (Distress → Resilience).

**Social Index** (higher = more resilient / stronger socio-economic position):
- 50% — Socio-Economic Score (CBS)
- 25% — Peripherality Index
- 25% — Income Support Rate (inverted)

**Health Index** (higher = healthier / lower disability burden):
- 50% — General Disability Rate (inverted)
- 25% — Special Services Disability Rate (inverted)
- 25% — Mobility Disability Rate (inverted)

---

## Data Sources

| Source | Data | Period |
|--------|------|--------|
| **Bituach Leumi** | Benefit recipients by settlement (>2,000 residents) | December 2024 |
| **CBS (LAMAS)** | Socio-Economic Index (cluster & score) | 2021 |
| **CBS (LAMAS)** | Peripherality Index (cluster & score) | 2020 |

---

## Project Structure

```
research-vicious-circle/
├── index.html                         # Redirect to presentation
├── presentation_main.html             # Interactive HTML presentation
├── generate_presentation_insights.py  # Script to regenerate presentation
├── research_vicious_circle.ipynb      # Main analysis notebook
```

> Note: raw data files are stored locally during development and are not necessarily committed to the repository.

---

## Notebooks

### Main Analysis
**`research_vicious_circle.ipynb`** — Full analysis pipeline with visualizations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Gop41_eXF3MNas1PvyZIGATu6VbdoP-o?usp=sharing)

### Clustering Analysis

---

## How to Update the Presentation

```bash
python generate_presentation_insights.py
```

This regenerates `presentation_main.html`. Push to GitHub and the live site updates automatically.

---

## Disclaimer

- This analysis shows **correlations**, not causation
- Administrative data reflects **benefit registration patterns**, which may differ from actual health conditions
- Access barriers and cultural factors affect who applies for benefits

---

## Links

- [Live Presentation](https://tovtechorg.github.io/research-vicious-circle/)
- [Project Kanban Board](https://docs.google.com/spreadsheets/d/1v2XxbDu7Gwhqv4Y4Dbebmn-HpkJRhlU1/edit?usp=sharing)
