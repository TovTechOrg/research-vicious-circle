# Vicious Circle Research

**Research Question:** Is there a "vicious circle" between socio-economic vulnerability and disability in Israel?

This project analyzes correlations between poverty indicators and health/disability outcomes across **278 Israeli settlements**, using administrative data from the National Insurance Institute (Bituach Leumi) and the Central Bureau of Statistics.

---

## Live Presentation

**[View Interactive Presentation](https://tovtechorg.github.io/research-vicious-circle/)**

The presentation includes:
- Interactive map of Israel (filter by SE cluster, disability rate, income support)
- Social Index vs Health Index correlation (Spearman r = 0.57)
- Resilience vs Distress analysis (outlier detection)
- Intergenerational Trap visualization (adult vs child disability)
- Detailed appendix with supporting correlations

---

## Key Findings

| Finding | Description |
|---------|-------------|
| **Vicious Circle Confirmed** | Social vulnerability and disability rates are strongly correlated (r = 0.57) |
| **Geography Matters** | Poverty in the periphery (North) is "more toxic" than poverty in the Center |
| **Intergenerational Risk** | High adult disability often co-exists with high child disability in the same localities |
| **Non-Linear Pattern** | Disability rates drop sharply only in the wealthiest quartile (Q4) |

---

## Data Sources

| Source | Data | Period |
|--------|------|--------|
| **Bituach Leumi** | Benefit recipients by settlement (>2,000 residents) | December 2024 |
| **CBS (LAMAS)** | Socio-Economic Index (cluster & score) | 2021 |
| **CBS (LAMAS)** | Peripherality Index (cluster & score) | 2020 |

---

## Methodology

### Social Index (X-axis)
Measures socio-economic vulnerability (higher = more vulnerable):
- 50% — Socio-Economic Score (inverted)
- 25% — Peripherality Index (inverted)
- 25% — Income Support Rate

### Health Index (Y-axis)
Measures disability burden (higher = worse health):
- 50% — General Disability Rate
- 25% — Special Services Disability Rate
- 25% — Mobility Disability Rate

Both indices are standardized and normalized to [0, 1].

---

## Project Structure

```
research-vicious-circle/
├── index.html                         # Redirect to presentation
├── presentation_main.html             # Interactive HTML presentation
├── generate_presentation_insights.py  # Script to regenerate presentation
├── research_vicious_circle (8).ipynb  # Main analysis notebook
├── vicious_circle_clustering.ipynb    # Clustering analysis
├── datas_for_research_vicious_circle_project/
│   ├── benefits_2024_12.xlsx          # Bituach Leumi data
│   ├── p_libud_23.xlsx                # CBS socio-economic data
│   └── ...                            # Additional data files
└── README.md
```

---

## Notebooks

### Main Analysis
**`research_vicious_circle (8).ipynb`** — Full analysis pipeline with visualizations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Gop41_eXF3MNas1PvyZIGATu6VbdoP-o?usp=sharing)

### Clustering Analysis
**`vicious_circle_clustering.ipynb`** — K-Means clustering and exploratory analysis

---

## How to Update the Presentation

```bash
# Edit insights/code in the generator script
# Then regenerate:
python generate_presentation_insights.py
```

This updates `presentation_main.html`. Push to GitHub and the live site updates automatically.

---

## Disclaimer

- This analysis shows **correlations**, not causality
- Administrative data reflects **benefit registration patterns**, which may differ from actual health conditions
- Access barriers and cultural factors affect who applies for benefits

---

## Links

- [Live Presentation](https://tovtechorg.github.io/research-vicious-circle/)
- [Project Kanban Board](https://docs.google.com/spreadsheets/d/1v2XxbDu7Gwhqv4Y4Dbebmn-HpkJRhlU1/edit?usp=sharing)
