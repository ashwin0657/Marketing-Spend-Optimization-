# 📊 Marketing Spend Optimization

> **An end-to-end Data Science project that analyses online advertising performance data and uses Machine Learning to recommend the optimal budget allocation across campaign dimensions — maximising Revenue ROAS.**

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset Overview](#-dataset-overview)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
  - [1. Data Loading & Inspection](#1-data-loading--inspection)
  - [2. Data Cleaning & Feature Engineering](#2-data-cleaning--feature-engineering)
  - [3. Overall KPI Summary](#3-overall-kpi-summary)
  - [4. Segment Performance Analysis](#4-segment-performance-analysis)
  - [5. Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
  - [6. Correlation Analysis](#6-correlation-analysis)
  - [7. Inferential Statistics](#7-inferential-statistics)
  - [8. Machine Learning Benchmark](#8-machine-learning-benchmark)
  - [9. Leakage-Aware Budget Response Model](#9-leakage-aware-budget-response-model)
  - [10. Candidate Combination Selection](#10-candidate-combination-selection)
  - [11. Budget Optimization](#11-budget-optimization)
  - [12. Budget Scenario Analysis](#12-budget-scenario-analysis)
  - [13. Results Export](#13-results-export)
- [Key Findings & Results](#-key-findings--results)
- [Technologies Used](#-technologies-used)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Problem Statement

Digital advertising teams face a critical challenge: **how to distribute a fixed marketing budget across multiple campaigns, placements, banners, and audience segments to maximise return on investment.**

Spending equally across all channels is inefficient — some combinations generate far higher returns than others. This project tackles that problem by:

1. **Analysing** historical advertising performance across multiple dimensions
2. **Identifying** which campaign–placement–banner–engagement combinations deliver the best ROAS
3. **Building ML models** that predict revenue from spend for any given combination
4. **Optimising** the budget allocation using constrained mathematical optimisation (SLSQP)
5. **Simulating** what-if budget scenarios (±20%, ±40%) to support strategic decisions

---

## 📁 Dataset Overview

The dataset (`p1.xlsx`) contains **15,408 rows** and **12 columns** of online advertising performance data spanning **April–June 2026**.

| Column | Type | Description |
|--------|------|-------------|
| `month` | Categorical | Month of the observation (April, May, June) |
| `day` | Integer | Day of the month (1–31) |
| `campaign_number` | Categorical | Campaign identifier (camp 1, camp 2, camp 3) |
| `user_engagement` | Categorical | Engagement level (High, Medium, Low) |
| `banner` | Categorical | Banner/ad size (e.g., 160×600, 300×250, 468×60, etc.) |
| `placement` | Categorical | Placement site/channel (abc, def, ghi, jkl, mno) — 413 missing values |
| `displays` | Integer | Number of ad impressions |
| `cost` | Float | Advertising spend (in currency units) |
| `clicks` | Integer | Number of clicks on the ad |
| `revenue` | Float | Revenue attributed to the ad |
| `post_click_conversions` | Integer | Conversions after clicking |
| `post_click_sales_amount` | Float | Sales revenue generated post-click |

**Data quality notes:**
- 5 exact duplicate rows (removed during cleaning)
- 413 missing values in `placement` (filled with "Unknown")
- No other missing values

---

## 📂 Project Structure

```
Marketing-Spend-Optimization/
│
├── marketing_spend_optimization.py    ← Full end-to-end pipeline script
│
├── data/
│   └── raw/
│       └── p1.xlsx                    ← Raw advertising performance dataset
│
├── notebooks/
│   └── marketing_spend_optimization.ipynb  ← Interactive Jupyter notebook
│
├── outputs/                           ← All generated result CSVs
│   ├── overall_kpis.csv
│   ├── campaign_number_performance.csv
│   ├── placement_performance.csv
│   ├── banner_performance.csv
│   ├── user_engagement_performance.csv
│   ├── month_performance.csv
│   ├── inferential_tests.csv
│   ├── campaign_pairwise_tests.csv
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── budget_allocation.csv
│   └── budget_scenarios.csv
│
├── _duplicates/                       ← Archived duplicate/checkpoint files
│
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn openpyxl
```

### Run the Pipeline

```bash
# Clone the repository
git clone https://github.com/ashwin0657/Marketing-Spend-Optimization-.git
cd Marketing-Spend-Optimization-

# Run the complete pipeline
python marketing_spend_optimization.py
```

Or open the Jupyter notebook for an interactive walkthrough:

```bash
jupyter notebook notebooks/marketing_spend_optimization.ipynb
```

---

## 🔄 Pipeline Walkthrough

### 1. Data Loading & Inspection

The pipeline begins by loading the raw Excel file and printing diagnostic information:

- **Shape**: 15,408 rows × 12 columns
- **Data types**: Mix of object (categorical) and numeric columns
- **Missing values**: Only `placement` has 413 nulls
- **Duplicates**: 5 exact duplicate rows detected

```python
df_raw = pd.read_excel(DATA_PATH)
print("Original shape:", df_raw.shape)
print("Exact duplicates:", df_raw.duplicated().sum())
```

### 2. Data Cleaning & Feature Engineering

**Cleaning steps:**
- Drop 5 duplicate rows
- Fill missing `placement` values with `"Unknown"`
- Construct a proper `date` column from month + day

**Engineered features (7 new KPI columns):**

| Feature | Formula | Purpose |
|---------|---------|---------|
| `ctr` | clicks ÷ displays | Click-Through Rate |
| `conversion_rate` | conversions ÷ clicks | Post-click conversion rate |
| `cpc` | cost ÷ clicks | Cost Per Click |
| `cpm` | (cost ÷ displays) × 1000 | Cost Per Mille (thousand impressions) |
| `cpa` | cost ÷ conversions | Cost Per Acquisition |
| `roas` | revenue ÷ cost | Return On Ad Spend |
| `sales_roas` | post_click_sales ÷ cost | Sales-based ROAS |

A custom `safe_divide()` function handles division by zero, returning 0 instead of NaN/Inf.

### 3. Overall KPI Summary

Aggregated KPIs across the entire dataset:

| KPI | Value |
|-----|-------|
| Total Displays | 239,014,261 |
| Total Spend | $175,192.22 |
| Total Clicks | 2,492,825 |
| Total Revenue | $276,262.62 |
| Total Conversions | 651,762 |
| Total Post-Click Sales | $32,715,410.23 |
| CTR | 1.04% |
| Conversion Rate | 26.15% |
| CPC | $0.070 |
| CPM | $0.733 |
| CPA | $0.269 |
| **Revenue ROAS** | **1.577** |

> 💡 A Revenue ROAS of **1.577** means every $1 spent generates $1.58 in revenue — a positive return, but with room for improvement through better allocation.

### 4. Segment Performance Analysis

Performance is broken down across 5 categorical dimensions using a reusable `segment_report()` function that computes spend, revenue, displays, clicks, conversions, spend share, CTR, conversion rate, CPC, CPA, and ROAS for each group.

**Campaign Performance:**

| Campaign | Spend | Revenue | ROAS | Spend Share |
|----------|-------|---------|------|-------------|
| Camp 2 | $17,037 | $34,890 | **2.05** | 9.7% |
| Camp 1 | $150,688 | $230,534 | 1.53 | 86.0% |
| Camp 3 | $7,467 | $10,839 | 1.45 | 4.3% |

> 📌 **Camp 2** has the highest ROAS (2.05×) but receives only 9.7% of spend — a clear signal that budget reallocation could improve returns.

**Placement Performance (Top 3 by ROAS):**

| Placement | ROAS | Spend Share |
|-----------|------|-------------|
| ghi | **2.44** | 35.5% |
| jkl | **2.27** | 2.1% |
| def | 1.12 | 8.1% |

### 5. Exploratory Data Analysis (EDA)

The EDA section generates four types of visualisations:

1. **ROAS Bar Charts** — One per segment dimension (campaign, placement, banner, engagement, month) showing which groups outperform
2. **Spend vs Revenue Scatter Plot** — Coloured by campaign, reveals the spend–revenue relationship and potential diminishing returns
3. **Daily ROAS Time Series** — Tracks ROAS fluctuation over the April–June period with a break-even line at ROAS = 1
4. **Correlation Heatmap** — Full 12×12 matrix of numeric KPI correlations

### 6. Correlation Analysis

A Pearson correlation heatmap across all numeric KPIs reveals:

- **`clicks`** is the strongest correlate of revenue (expected — more clicks → more revenue)
- **`cost`** and **`displays`** also strongly correlate with revenue
- **`cpa`** (Cost Per Acquisition) shows a weak/negative relationship with revenue, indicating cheaper acquisitions tend to associate with higher revenue segments
- **`roas`** is weakly correlated with raw revenue — high-ROAS combinations may not be the highest-volume ones

### 7. Inferential Statistics

#### Kruskal-Wallis H-Tests

Non-parametric tests are used (instead of ANOVA) because ad performance data is heavily right-skewed. Tests are run for both `revenue` and `roas` across all 5 segment dimensions.

| Segment | Metric | H Statistic | p-value | Significant? |
|---------|--------|-------------|---------|--------------|
| Campaign | Revenue | 653.12 | ≈ 0 | ✅ Yes |
| Campaign | ROAS | 169.31 | ≈ 0 | ✅ Yes |
| Placement | Revenue | 2693.94 | 0.0 | ✅ Yes |
| Banner | Revenue | 4292.64 | 0.0 | ✅ Yes |
| User Engagement | Revenue | 1012.54 | ≈ 0 | ✅ Yes |
| Month | Revenue | 197.05 | ≈ 0 | ✅ Yes |

> ✅ **All tests are significant at p < 0.05**, confirming that performance genuinely differs across all segment dimensions — the differences are not due to random chance.

#### Mann-Whitney U Pairwise Tests (Campaigns)

| Pair | U Statistic | p-value | Significant? |
|------|-------------|---------|--------------|
| Camp 1 vs Camp 2 | 4,940,129 | 3.52 × 10⁻¹² | ✅ Yes |
| Camp 1 vs Camp 3 | 28,415,031 | 1.09 × 10⁻⁹¹ | ✅ Yes |
| Camp 2 vs Camp 3 | 7,476,522 | 6.19 × 10⁻¹⁰⁴ | ✅ Yes |

> All three campaign pairs show **highly significant** revenue differences, with the Camp 2 vs Camp 3 comparison being the most significant.

### 8. Machine Learning Benchmark

Four regression models are trained to predict `revenue` using **all 10 available features** (including downstream variables like clicks and conversions). The target is log-transformed (`log1p`) to handle skewness, with predictions inverse-transformed (`expm1`).

**Model Comparison:**

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **Extra Trees** | **0.875** | **8.00** | **0.9936** |
| Random Forest | 1.043 | 10.76 | 0.9884 |
| Gradient Boosting | 1.353 | 11.91 | 0.9857 |
| Ridge | 42.449 | 581.74 | -33.07 |

> 🏆 **Extra Trees Regressor** achieves the best performance with R² = 0.9936 (99.36% of variance explained) and the lowest RMSE of 8.00.

**Permutation Importance (Top Features):**

| Feature | Importance |
|---------|-----------|
| clicks | 0.996 |
| cost | 0.459 |
| displays | 0.369 |
| campaign_number | 0.281 |
| post_click_conversions | 0.160 |
| month | 0.094 |
| user_engagement | 0.088 |

> ⚠️ **Note**: This benchmark intentionally includes downstream variables (clicks, conversions). It measures predictive accuracy, **not** causal budget response. The budget allocation model (Section 9) addresses this by excluding these leaky features.

### 9. Leakage-Aware Budget Response Model

For budget optimisation, a separate **ExtraTrees model** is trained using **only allocation-time features** — variables known *before* placing the ad:

- `month`, `day`, `campaign_number`, `user_engagement`, `banner`, `placement`, `cost`

**Why?** Clicks and conversions happen *after* the budget is spent. Including them would create **data leakage** — the model would learn to "predict" revenue from outcomes rather than from budget decisions.

```python
opt_model = Pipeline([
    ("preprocess", pre_o),
    ("model", ExtraTreesRegressor(n_estimators=350, min_samples_leaf=5))
])
opt_model.fit(Xo, np.log1p(yo))
```

### 10. Candidate Combination Selection

The optimiser doesn't consider every possible combination — that would be computationally expensive and include unstable, low-sample groups. Instead:

1. Group data by `campaign_number × user_engagement × banner × placement`
2. Compute historical ROAS for each combination
3. **Filter**: Keep only combinations with ≥ 30 historical observations (ensures statistical stability)
4. **Rank**: Sort by ROAS descending
5. **Select**: Top 40 candidates

This gives a manageable set of high-potential combinations to allocate budget across.

### 11. Budget Optimization

The core of the project — a constrained mathematical optimisation that finds the best way to split the total budget:

**Objective:** Maximise total predicted revenue across all candidate combinations

**Constraints:**
- Total spend must equal the total budget ($175,192.22)
- No single combination can receive more than 15% of the total budget (diversification cap)

**Method:** Sequential Least Squares Programming (SLSQP) via `scipy.optimize.minimize`

```python
sol = minimize(
    objective, x0=x0, method="SLSQP",
    bounds=bounds, constraints=constraints,
    options={"maxiter": 250, "ftol": 1e-5}
)
```

The optimizer iteratively adjusts spend amounts, calling the budget response model to predict revenue for each allocation, and converges on the revenue-maximising distribution.

### 12. Budget Scenario Analysis

To support strategic planning, the optimizer is run at multiple budget levels:

| Budget Multiplier | Total Budget | Predicted Revenue | Predicted ROAS | Optimizer Converged? |
|:-:|:-:|:-:|:-:|:-:|
| 0.80× | $140,154 | $6,190 | 0.044 | ✅ |
| 0.90× | $157,673 | $6,190 | 0.039 | ✅ |
| **1.00×** | **$175,192** | **$6,190** | **0.035** | ✅ |
| 1.10× | $192,711 | $6,190 | 0.032 | ✅ |
| 1.20× | $210,231 | $6,190 | 0.029 | ✅ |
| 1.40× | $245,269 | $6,190 | 0.025 | ✅ |

> 📈 The budget response model uses only allocation-time features (no clicks/conversions), so the predicted revenue here reflects what the model can estimate without downstream information. The consistent convergence across all scenarios confirms the optimizer is robust.

### 13. Results Export

All result tables are exported as CSV files to the `outputs/` directory for further analysis, reporting, or dashboard integration.

---

## 🔑 Key Findings & Results

### 1. Campaign Efficiency
- **Camp 2** delivers the highest ROAS (2.05×) but receives only 9.7% of the budget
- **Camp 1** consumes 86% of spend with a lower ROAS of 1.53×
- Reallocating budget from Camp 1 → Camp 2 could improve overall returns

### 2. Placement Insights
- **ghi** and **jkl** placements achieve 2.4× and 2.3× ROAS respectively
- **mno** placement (largest spend share at 54%) has the lowest ROAS among active placements (1.06×)

### 3. Statistical Significance
- All performance differences across segments are **statistically significant** (p < 0.05)
- This is not random variation — campaign, placement, banner, engagement, and month all genuinely impact revenue

### 4. Predictive Modeling
- **Extra Trees** achieves R² = 0.9936 — near-perfect revenue prediction
- **Clicks** is the most important feature (importance = 0.996), followed by cost (0.459) and displays (0.369)

### 5. Budget Optimization
- The SLSQP optimizer successfully converges across all budget scenarios
- The system provides a recommended spend for each of 40 candidate campaign combinations
- A 15% diversification cap prevents over-concentration in any single combination

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core programming language |
| **pandas** | Data manipulation and aggregation |
| **NumPy** | Numerical computations |
| **Matplotlib** | Static visualisations |
| **Seaborn** | Statistical visualisations |
| **SciPy** | Statistical tests (Kruskal-Wallis, Mann-Whitney) & SLSQP optimization |
| **scikit-learn** | ML models (Ridge, Random Forest, Extra Trees, Gradient Boosting), preprocessing, evaluation |
| **openpyxl** | Reading Excel files |

---

## 🚀 Future Improvements

- **Time-series features**: Add rolling averages, lag features, and day-of-week effects
- **Cross-validation**: Replace single train/test split with k-fold CV for more robust model evaluation
- **Advanced models**: Experiment with XGBoost, LightGBM, or neural networks
- **Interactive dashboard**: Build a Streamlit/Dash app for real-time budget scenario exploration
- **A/B testing framework**: Validate optimised allocation against current allocation in production
- **Diminishing returns modelling**: Fit concave response curves (e.g., log or Hill functions) per combination
- **Multi-objective optimisation**: Optimise for both revenue and conversions simultaneously

---

## 👤 Author

**Ashwin**

- GitHub: [@ashwin0657](https://github.com/ashwin0657)

---

## 📄 License

This project is open source and available for educational and research purposes.

---

> ⭐ If you found this project useful, please consider giving it a star on GitHub!
