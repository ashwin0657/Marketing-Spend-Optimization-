
# Marketing Spend Optimization - End-to-End Data Science Project
# Dataset: p1.xlsx

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
DATA_PATH = Path("p1.xlsx")
if not DATA_PATH.exists():
    DATA_PATH = Path("/mnt/data/p1.xlsx")

# ============================================================
# 1. LOAD DATA
# ============================================================
df_raw = pd.read_excel(DATA_PATH)
print("Original shape:", df_raw.shape)
print(df_raw.head())
print("\nData types:\n", df_raw.dtypes)
print("\nMissing values:\n", df_raw.isna().sum())
print("\nExact duplicates:", df_raw.duplicated().sum())

# ============================================================
# 2. CLEANING + FEATURE ENGINEERING
# ============================================================
df = df_raw.drop_duplicates().copy()
df["placement"] = df["placement"].fillna("Unknown")

month_order = {"April": 4, "May": 5, "June": 6}
df["month_num"] = df["month"].map(month_order)
df["date"] = pd.to_datetime(
    "2026-" + df["month_num"].astype(str) + "-" + df["day"].astype(str),
    errors="coerce"
)

def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros(len(a), dtype=float), where=np.asarray(b) != 0)

df["ctr"] = safe_divide(df["clicks"], df["displays"])
df["conversion_rate"] = safe_divide(df["post_click_conversions"], df["clicks"])
df["cpc"] = safe_divide(df["cost"], df["clicks"])
df["cpm"] = safe_divide(df["cost"], df["displays"]) * 1000
df["cpa"] = safe_divide(df["cost"], df["post_click_conversions"])
df["roas"] = safe_divide(df["revenue"], df["cost"])
df["sales_roas"] = safe_divide(df["post_click_sales_amount"], df["cost"])

# ============================================================
# 3. OVERALL KPIs
# ============================================================
totals = df[
    ["displays","cost","clicks","revenue",
     "post_click_conversions","post_click_sales_amount"]
].sum()

overall_kpis = pd.Series({
    "Total displays": totals["displays"],
    "Total spend": totals["cost"],
    "Total clicks": totals["clicks"],
    "Total revenue": totals["revenue"],
    "Total conversions": totals["post_click_conversions"],
    "Total post-click sales": totals["post_click_sales_amount"],
    "CTR": totals["clicks"] / totals["displays"],
    "Conversion rate": totals["post_click_conversions"] / totals["clicks"],
    "CPC": totals["cost"] / totals["clicks"],
    "CPM": totals["cost"] / totals["displays"] * 1000,
    "CPA": totals["cost"] / totals["post_click_conversions"],
    "Revenue ROAS": totals["revenue"] / totals["cost"],
})
print("\nOVERALL KPIs\n", overall_kpis)

# ============================================================
# 4. SEGMENT PERFORMANCE
# ============================================================
def segment_report(data, group_col):
    g = data.groupby(group_col, dropna=False).agg(
        spend=("cost","sum"),
        revenue=("revenue","sum"),
        displays=("displays","sum"),
        clicks=("clicks","sum"),
        conversions=("post_click_conversions","sum"),
        sales=("post_click_sales_amount","sum"),
        rows=("cost","size")
    )
    g["spend_share"] = g["spend"] / g["spend"].sum()
    g["ctr"] = safe_divide(g["clicks"].values, g["displays"].values)
    g["conversion_rate"] = safe_divide(g["conversions"].values, g["clicks"].values)
    g["cpc"] = safe_divide(g["spend"].values, g["clicks"].values)
    g["cpa"] = safe_divide(g["spend"].values, g["conversions"].values)
    g["roas"] = safe_divide(g["revenue"].values, g["spend"].values)
    return g.sort_values("roas", ascending=False)

reports = {}
for col in ["campaign_number","placement","banner","user_engagement","month"]:
    reports[col] = segment_report(df, col)
    print("\n================", col.upper(), "================")
    print(reports[col].round(4).to_string())

# ============================================================
# 5. EDA
# ============================================================
def barplot_report(report, group_name, metric="roas"):
    tmp = report.reset_index()
    plt.figure(figsize=(10,5))
    sns.barplot(data=tmp, x=group_name, y=metric)
    plt.title(f"{metric.upper()} by {group_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

for col in ["campaign_number","placement","banner","user_engagement","month"]:
    barplot_report(reports[col], col)

plt.figure(figsize=(9,6))
sns.scatterplot(
    data=df, x="cost", y="revenue",
    hue="campaign_number", alpha=0.45
)
plt.title("Advertising Spend vs Revenue")
plt.tight_layout()
plt.show()

daily = df.groupby("date").agg(
    spend=("cost","sum"),
    revenue=("revenue","sum")
)
daily["roas"] = daily["revenue"] / daily["spend"]

plt.figure(figsize=(12,5))
plt.plot(daily.index, daily["roas"])
plt.axhline(1, linestyle="--")
plt.title("Daily Revenue ROAS")
plt.ylabel("ROAS")
plt.tight_layout()
plt.show()

# ============================================================
# 6. CORRELATION
# ============================================================
corr_cols = [
    "displays","cost","clicks","revenue",
    "post_click_conversions","post_click_sales_amount",
    "ctr","conversion_rate","cpc","cpm","cpa","roas"
]
corr = df[corr_cols].corr()

plt.figure(figsize=(12,9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Marketing KPI Correlation Matrix")
plt.tight_layout()
plt.show()

print("\nCorrelation with revenue:")
print(corr["revenue"].sort_values(ascending=False))

# ============================================================
# 7. INFERENTIAL STATISTICS
# ============================================================
def kruskal_test(data, group_col, metric):
    groups = [
        x[metric].dropna().values
        for _, x in data.groupby(group_col)
        if len(x[metric].dropna()) > 1
    ]
    h, p = stats.kruskal(*groups)
    return [group_col, metric, h, p, p < 0.05]

tests = []
for col in ["campaign_number","placement","banner","user_engagement","month"]:
    tests.append(kruskal_test(df, col, "revenue"))
    tests.append(kruskal_test(df, col, "roas"))

tests_df = pd.DataFrame(
    tests,
    columns=["group","metric","H_statistic","p_value","significant_5pct"]
)
print("\nKRUSKAL-WALLIS TESTS")
print(tests_df.to_string(index=False))

# Campaign pairwise test
campaigns = sorted(df["campaign_number"].unique())
pairs = []
for i in range(len(campaigns)):
    for j in range(i+1, len(campaigns)):
        a = df.loc[df.campaign_number == campaigns[i], "revenue"]
        b = df.loc[df.campaign_number == campaigns[j], "revenue"]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        pairs.append([campaigns[i], campaigns[j], u, p])

pairs_df = pd.DataFrame(
    pairs, columns=["group1","group2","U","p_value"]
)
print("\nCAMPAIGN PAIRWISE MANN-WHITNEY TEST")
print(pairs_df.to_string(index=False))

# ============================================================
# 8. ML BENCHMARK
# ============================================================
# This benchmark intentionally includes downstream variables.
# It is a predictive model, not the budget-allocation model.

target = "revenue"
benchmark_features = [
    "month","day","campaign_number","user_engagement","banner",
    "placement","displays","cost","clicks","post_click_conversions"
]

X = df[benchmark_features]
y = df[target]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

models = {
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=250, random_state=RANDOM_STATE,
        n_jobs=-1, min_samples_leaf=2
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=250, random_state=RANDOM_STATE,
        n_jobs=-1, min_samples_leaf=2
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=RANDOM_STATE, n_estimators=200
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

results = []
fitted = {}

for name, model in models.items():
    pipe = Pipeline([("preprocess", pre), ("model", model)])
    pipe.fit(X_train, np.log1p(y_train))
    pred = np.expm1(pipe.predict(X_test)).clip(min=0)

    results.append([
        name,
        mean_absolute_error(y_test, pred),
        mean_squared_error(y_test, pred) ** 0.5,
        r2_score(y_test, pred)
    ])
    fitted[name] = pipe

model_results = pd.DataFrame(
    results, columns=["model","MAE","RMSE","R2"]
).sort_values("RMSE")

print("\nMODEL COMPARISON")
print(model_results.to_string(index=False))

best_name = model_results.iloc[0]["model"]
best_pipe = fitted[best_name]

perm = permutation_importance(
    best_pipe, X_test, np.log1p(y_test),
    n_repeats=5, random_state=RANDOM_STATE,
    scoring="neg_root_mean_squared_error"
)

importance = pd.Series(
    perm.importances_mean, index=X_test.columns
).sort_values(ascending=False)

print("\nPERMUTATION IMPORTANCE")
print(importance)

# ============================================================
# 9. LEAKAGE-AWARE BUDGET RESPONSE MODEL
# ============================================================
# Allocation-time features only + spend.
# Clicks and conversions are excluded because they occur downstream.

opt_features = [
    "month","day","campaign_number",
    "user_engagement","banner","placement","cost"
]

Xo = df[opt_features]
yo = df["revenue"]

cat_o = Xo.select_dtypes(include="object").columns.tolist()
num_o = [c for c in Xo.columns if c not in cat_o]

pre_o = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_o),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_o)
])

opt_model = Pipeline([
    ("preprocess", pre_o),
    ("model", ExtraTreesRegressor(
        n_estimators=350, random_state=RANDOM_STATE,
        n_jobs=-1, min_samples_leaf=5
    ))
])

opt_model.fit(Xo, np.log1p(yo))

# ============================================================
# 10. CANDIDATE COMBINATIONS
# ============================================================
combo_cols = [
    "campaign_number","user_engagement","banner","placement"
]

combo_stats = df.groupby(combo_cols).agg(
    rows=("cost","size"),
    spend=("cost","sum"),
    revenue=("revenue","sum"),
    median_cost=("cost","median")
).reset_index()

combo_stats["roas"] = combo_stats["revenue"] / combo_stats["spend"]

# Minimum historical sample protects against tiny, unstable combinations.
candidates = (
    combo_stats.query("rows >= 30")
    .sort_values("roas", ascending=False)
    .head(40)
    .reset_index(drop=True)
)

print("\nTOP CANDIDATE COMBINATIONS")
print(candidates.head(15).to_string(index=False))

# ============================================================
# 11. BUDGET OPTIMIZATION
# ============================================================
scenario_month = "June"
scenario_day = 15
TOTAL_BUDGET = float(df["cost"].sum())

def predict_revenue_for_spend(combo_row, spend):
    row = {
        "month": scenario_month,
        "day": scenario_day,
        "campaign_number": combo_row["campaign_number"],
        "user_engagement": combo_row["user_engagement"],
        "banner": combo_row["banner"],
        "placement": combo_row["placement"],
        "cost": max(float(spend), 0.0)
    }
    return float(
        np.expm1(opt_model.predict(pd.DataFrame([row]))[0]).clip(0)
    )

def optimize_budget(total_budget, max_share=0.15):
    cand = candidates.copy()

    if len(cand) * max_share < 1:
        cand = cand.head(int(np.ceil(1 / max_share))).copy()

    n = len(cand)
    x0 = np.full(n, total_budget / n)
    bounds = [(0, total_budget * max_share)] * n

    def objective(x):
        return -sum(
            predict_revenue_for_spend(cand.iloc[i], x[i])
            for i in range(n)
        )

    constraints = [{
        "type": "eq",
        "fun": lambda x: np.sum(x) - total_budget
    }]

    sol = minimize(
        objective, x0=x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 250, "ftol": 1e-5}
    )

    allocation = cand.copy()
    allocation["recommended_spend"] = sol.x
    allocation["predicted_revenue"] = [
        predict_revenue_for_spend(cand.iloc[i], sol.x[i])
        for i in range(n)
    ]
    allocation["predicted_roas"] = (
        allocation["predicted_revenue"] /
        allocation["recommended_spend"].replace(0, np.nan)
    )
    allocation = allocation.sort_values(
        "recommended_spend", ascending=False
    )

    return allocation, sol

allocation, solution = optimize_budget(TOTAL_BUDGET)

print("\nOPTIMIZER STATUS:", solution.success, solution.message)
print("\nRECOMMENDED BUDGET ALLOCATION")
print(
    allocation[
        combo_cols + [
            "rows","roas",
            "recommended_spend",
            "predicted_revenue",
            "predicted_roas"
        ]
    ].head(20).to_string(index=False)
)

pred_total = allocation["predicted_revenue"].sum()
spend_total = allocation["recommended_spend"].sum()

print("\nOptimized budget:", spend_total)
print("Predicted revenue:", pred_total)
print("Predicted ROAS:", pred_total / spend_total)

# ============================================================
# 12. BUDGET SCENARIOS
# ============================================================
scenario_rows = []
for multiplier in [0.80, 0.90, 1.00, 1.10, 1.20, 1.40]:
    alloc_s, sol_s = optimize_budget(TOTAL_BUDGET * multiplier)
    budget = alloc_s["recommended_spend"].sum()
    revenue = alloc_s["predicted_revenue"].sum()
    scenario_rows.append([
        multiplier, budget, revenue, revenue / budget, sol_s.success
    ])

scenario_results = pd.DataFrame(
    scenario_rows,
    columns=["budget_multiplier","budget","predicted_revenue",
             "predicted_roas","optimizer_success"]
)

print("\nBUDGET SCENARIOS")
print(scenario_results.to_string(index=False))

# ============================================================
# 13. EXPORT RESULTS
# ============================================================
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

overall_kpis.to_csv(OUT / "overall_kpis.csv")
model_results.to_csv(OUT / "model_comparison.csv", index=False)
tests_df.to_csv(OUT / "inferential_tests.csv", index=False)
pairs_df.to_csv(OUT / "campaign_pairwise_tests.csv", index=False)
importance.to_csv(OUT / "feature_importance.csv")

for col, report in reports.items():
    report.to_csv(OUT / f"{col}_performance.csv")

allocation.to_csv(OUT / "budget_allocation.csv", index=False)
scenario_results.to_csv(OUT / "budget_scenarios.csv", index=False)

print("\nAll result tables exported to:", OUT.resolve())
