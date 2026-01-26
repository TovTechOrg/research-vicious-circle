# Vicious Circle Research

An interactive HTML presentation that explores correlations between **socio-economic vulnerability** and **health / disability outcomes** across Israeli settlements.

## Live Presentation (GitHub Pages)

- Main URL: https://tovtechorg.github.io/research-vicious-circle/
- Direct file: https://tovtechorg.github.io/research-vicious-circle/presentation_main.html

## What’s Inside

The presentation includes (among others):

- Interactive map of Israel with a dropdown (Socio‑Economic Cluster / General Disability Rate / Income Support Rate)
- Composite **Social Index** vs **Health Index** correlation (Spearman shown on slide)
- Residuals / outliers (“Resilience vs. Distress”) to highlight “break-the-rules” localities
- Spearman correlation heatmap between key social & health indicators
- “Intergenerational Trap” (adult disability vs child disability) with a Red Zone defined by national weighted averages
- Appendix section with additional supporting correlations

## High‑Level Findings (From the Slides)

- Social vulnerability and disability severity move together across localities (composite indices show a clear monotonic relationship).
- Residual analysis highlights localities with “excess disability” vs. “resilience” relative to the model (potential regional hubs vs. strong community buffers).
- The intergenerational pattern suggests a risk of a self‑reinforcing cycle where high adult disability co‑exists with high child disability in the same places.
- Disability rates by socio‑economic quartile are **non‑linear**: the highest quartile drops sharply compared to the lower three quartiles (for settlements >10k).

## Data Sources (Unified Dataset)

- **National Insurance Institute (Bituach Leumi)**: administrative records, benefit recipients in localities with **>2,000 residents** (current as of **December 2024**)
- **Central Bureau of Statistics (CBS)**:
  - Socio‑Economic Index (Cluster & Score, updated to **2021**)
  - Peripherality Index (Cluster & Score, updated to **2020**)

## Methodology (Core Indices)

Both indices are normalized to **[-1, +1]** (Distress → Resilience).

- **Social Index (X‑Axis)** (weighted):
  - 50% Socio‑Economic Score (CBS)
  - 25% Peripherality Index
  - 25% Income Support Rate (inverted)
- **Health Index (Y‑Axis)** (weighted, inverted so “higher = healthier”):
  - 50% General Disability Rate
  - 25% Special Services Disability Rate
  - 25% Mobility Disability Rate

## How to Update the Presentation

The HTML is generated from the project data with the generator script:

1) Update code/insights in `generate_presentation_insights.py`
2) Regenerate:

```bash
python generate_presentation_insights.py
```

This updates `presentation_main.html`. The `index.html` file redirects to it (so Pages opens the presentation by default).

## Repository Layout

- `generate_presentation_insights.py` — main generator for the presentation
- `presentation_main.html` — generated output (served by GitHub Pages)
- `index.html` — redirect entrypoint for Pages
- `datas_for_research_vicious_circle_project/` — project dataset files

## Notes / Disclaimer

- The analysis highlights **correlations**; it does not prove causality.
- Administrative records can reflect **access/registration patterns** as well as underlying conditions.

## Links

- Task Board (Kanban): https://docs.google.com/spreadsheets/d/1v2XxbDu7Gwhqv4Y4Dbebmn-HpkJRhlU1/edit?usp=sharing
