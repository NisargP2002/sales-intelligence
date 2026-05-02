import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import DB_URL, OUTPUT_DIR

engine = create_engine(DB_URL)
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False

# ── Load from PostgreSQL ─────────────────────────────────────

df = pd.read_sql("""
    SELECT
        f.*, p.category, p.sub_category, p.product_name,
        c.customer_name, c.segment,
        l.city, l.state, l.region
    FROM fact_sales f
    JOIN dim_product  p ON f.product_id  = p.product_id
    JOIN dim_customer c ON f.customer_id = c.customer_id
    JOIN dim_location l ON f.location_id = l.location_id
""", engine)

print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── Plot 1: Revenue trend ────────────────────────────────────

monthly = (
    df.groupby(["order_year", "order_month"])["sales"]
    .sum()
    .reset_index()
)
monthly["date"] = pd.to_datetime(
    monthly[["order_year", "order_month"]]
    .assign(day=1)
    .rename(columns={"order_year": "year",
                     "order_month": "month"})
)
monthly = monthly.sort_values("date")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(monthly["date"], monthly["sales"],
        color="#534AB7", linewidth=2)
ax.fill_between(monthly["date"], monthly["sales"],
                alpha=0.1, color="#534AB7")
ax.set_xlabel("Date")
ax.set_ylabel("Monthly Revenue ($)")
ax.set_title("Revenue trend 2014-2017")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "revenue_trend.png", dpi=150)
plt.close()
print(f"Plot 1 saved — revenue_trend.png")
print(f"  Best month  : ${monthly['sales'].max():,.0f}")
print(f"  Worst month : ${monthly['sales'].min():,.0f}")
print(f"  Growth      : {((monthly['sales'].iloc[-1] / monthly['sales'].iloc[0]) - 1):.1%}")

# ── Plot 2: Revenue by category ──────────────────────────────

cat = df.groupby("category").agg(
    revenue=("sales", "sum"),
    profit=("profit", "sum"),
    orders=("order_id", "nunique")
).reset_index()
cat["margin"] = (cat["profit"] / cat["revenue"] * 100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(cat["category"], cat["revenue"],
            color=["#534AB7", "#1D9E75", "#D85A30"])
axes[0].set_title("Revenue by category")
axes[0].set_ylabel("Revenue ($)")

axes[1].bar(cat["category"], cat["margin"],
            color=["#534AB7", "#1D9E75", "#D85A30"])
axes[1].axhline(0, color="#444441", linewidth=0.8)
axes[1].set_title("Profit margin % by category")
axes[1].set_ylabel("Margin %")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "category_analysis.png", dpi=150)
plt.close()
print(f"\nPlot 2 saved — category_analysis.png")
print(cat[["category", "revenue", "margin"]].to_string(index=False))

# ── Plot 3: Region performance ───────────────────────────────

region = df.groupby("region").agg(
    revenue=("sales", "sum"),
    profit=("profit", "sum"),
    orders=("order_id", "nunique"),
    customers=("customer_id", "nunique")
).reset_index().sort_values("revenue", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bars = axes[0].bar(region["region"], region["revenue"],
                   color="#1D9E75")
axes[0].bar_label(
    bars,
    labels=[f"${v/1000:.0f}k" for v in region["revenue"]],
    padding=4
)
axes[0].set_title("Revenue by region")

axes[1].scatter(region["revenue"], region["profit"],
                s=region["orders"] * 2,
                color="#534AB7", alpha=0.7)
for _, row in region.iterrows():
    axes[1].annotate(
        row["region"],
        (row["revenue"], row["profit"]),
        fontsize=10, ha="center", va="bottom"
    )
axes[1].set_xlabel("Revenue")
axes[1].set_ylabel("Profit")
axes[1].set_title("Revenue vs Profit by region")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "region_analysis.png", dpi=150)
plt.close()
print(f"\nPlot 3 saved — region_analysis.png")
print(region[["region", "revenue", "profit"]].to_string(index=False))

# ── Plot 4: Discount impact ──────────────────────────────────

bins   = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
labels = ["0-10%", "10-20%", "20-30%",
          "30-40%", "40-50%", "50%+"]
df["discount_bin"] = pd.cut(
    df["discount"], bins=bins,
    labels=labels, include_lowest=True
)

disc = df.groupby("discount_bin").agg(
    avg_profit_margin=("profit_margin", "mean"),
    order_count=("order_id", "count")
).reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#1D9E75" if v >= 0 else "#D85A30"
          for v in disc["avg_profit_margin"]]
ax.bar(disc["discount_bin"],
       disc["avg_profit_margin"] * 100, color=colors)
ax.axhline(0, color="#444441", linewidth=1)
ax.set_xlabel("Discount tier")
ax.set_ylabel("Avg profit margin %")
ax.set_title("Discount impact on margin")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "discount_analysis.png", dpi=150)
plt.close()
print(f"\nPlot 4 saved — discount_analysis.png")

# ── Plot 5: Top 10 sub-categories by revenue ─────────────────

subcat = (
    df.groupby("sub_category")["sales"]
    .sum()
    .sort_values(ascending=True)
    .tail(10)
)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(subcat.index, subcat.values, color="#534AB7", alpha=0.8)
ax.set_xlabel("Revenue ($)")
ax.set_title("Top 10 sub-categories by revenue")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "subcategory_analysis.png", dpi=150)
plt.close()
print(f"Plot 5 saved — subcategory_analysis.png")

# ── EDA summary ──────────────────────────────────────────────

print("""
=== EDA INSIGHTS ===

1. Revenue is seasonal — Q4 spikes every year
   Action: quarter + month as forecast features

2. Technology highest revenue, Office Supplies highest margin
   Furniture near-zero or negative margin
   Action: category as critical forecast feature

3. West leads revenue, South lags
   Action: region encoding in forecast model

4. Discounts above 20% destroy margin
   30%+ discount = negative profit on average
   Action: discount as forecast feature

5. Consumer segment drives most volume
   Corporate has better margin per order
   Action: segment encoding in forecast model
""")

engine.dispose()
print("EDA complete. Check outputs/ for all plots.")