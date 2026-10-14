# Marketing Spend Optimization

End-to-end data science project for optimising digital advertising budget allocation.

## Project Structure

```
├── marketing_spend_optimization.py  ← Full pipeline script
├── data/
│   └── raw/
│       └── p1.xlsx                  ← Raw advertising performance dataset
├── notebooks/
│   └── marketing_spend_optimization.ipynb  ← Interactive notebook
├── outputs/                         ← Generated result CSVs
├── _duplicates/                     ← Old/duplicate files (safe archive)
└── .gitignore
```

## Quick Start

```bash
python marketing_spend_optimization.py
```

Or open the notebook in `notebooks/` for an interactive walkthrough.

## Pipeline Stages

1. **Data Loading** — Read `p1.xlsx`, inspect shape, types, missing values
2. **Cleaning & Feature Engineering** — Deduplicate, impute, compute CTR/CPC/ROAS etc.
3. **Overall KPIs** — Aggregate totals and efficiency ratios
4. **Segment Performance** — Break down by campaign, placement, banner, engagement, month
5. **EDA** — Bar plots, scatter plots, daily ROAS trend, correlation heatmap
6. **Inferential Statistics** — Kruskal-Wallis + Mann-Whitney pairwise tests
7. **ML Benchmark** — Ridge, Random Forest, Extra Trees, Gradient Boosting comparison
8. **Budget Response Model** — Leakage-aware model using only allocation-time features
9. **Budget Optimization** — SLSQP constrained optimisation across candidate combos
10. **Budget Scenarios** — Evaluate 80%–140% budget multipliers
11. **Export** — All tables saved to `outputs/`

## Dependencies

- Python 3.10+
- pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, openpyxl
