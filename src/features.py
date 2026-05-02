import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine

sys.path.append(str(Path(__file__).parent.parent))
from config import DB_URL, PROC_DIR, RANDOM_STATE, TEST_SIZE


def load_from_warehouse():
    engine = create_engine(DB_URL)
    df = pd.read_sql("""
        SELECT
            f.order_date, f.sales, f.profit,
            f.quantity, f.discount, f.ship_days,
            f.profit_margin, f.order_year,
            f.order_month, f.order_quarter,
            p.category, p.sub_category,
            c.segment,
            l.region
        FROM fact_sales f
        JOIN dim_product  p ON f.product_id  = p.product_id
        JOIN dim_customer c ON f.customer_id = c.customer_id
        JOIN dim_location l ON f.location_id = l.location_id
    """, engine)
    engine.dispose()
    df["order_date"] = pd.to_datetime(df["order_date"])
    print(f"Loaded: {df.shape[0]} rows from warehouse")
    return df


def build_daily_series(df):
    """
    Aggregate to daily revenue — the unit the model forecasts.
    One row = one day = total sales across all orders that day.
    """
    daily = (
        df.groupby("order_date")
        .agg(
            revenue        = ("sales",   "sum"),
            profit         = ("profit",  "sum"),
            orders         = ("sales",   "count"),
            avg_discount   = ("discount","mean"),
            avg_ship_days  = ("ship_days","mean"),
            qty_sold       = ("quantity", "sum")
        )
        .reset_index()
        .sort_values("order_date")
    )

    # Fill missing dates with 0 — business was open every day
    full_range = pd.date_range(
        daily["order_date"].min(),
        daily["order_date"].max(),
        freq="D"
    )
    daily = (
        daily.set_index("order_date")
        .reindex(full_range, fill_value=0)
        .rename_axis("order_date")
        .reset_index()
    )
    print(f"Daily series: {len(daily)} days")
    return daily


def add_time_features(df):
    """
    Insight 1 from EDA: revenue is seasonal.
    Time features let the model learn Q4 spikes.
    """
    df = df.copy()
    df["year"]        = df["order_date"].dt.year
    df["month"]       = df["order_date"].dt.month
    df["quarter"]     = df["order_date"].dt.quarter
    df["day_of_week"] = df["order_date"].dt.dayofweek
    df["day_of_year"] = df["order_date"].dt.dayofyear
    df["week"]        = df["order_date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_month_end"]   = df["order_date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["order_date"].dt.is_month_start.astype(int)
    df["is_q4"]       = (df["quarter"] == 4).astype(int)

    print(f"  Added 10 time features")
    return df


def add_lag_features(df):
    """
    Lag features tell the model what revenue looked like
    in the recent past — the most powerful forecast signal.
    """
    df = df.copy()
    lags = [1, 7, 14, 30]
    for lag in lags:
        df[f"revenue_lag_{lag}"] = df["revenue"].shift(lag)

    print(f"  Added {len(lags)} lag features: {lags}")
    return df


def add_rolling_features(df):
    """
    Rolling averages smooth out noise and capture trends.
    7-day = weekly pattern. 30-day = monthly trend.
    """
    df = df.copy()
    windows = [7, 14, 30]
    for w in windows:
        df[f"revenue_rolling_mean_{w}"] = (
            df["revenue"].shift(1).rolling(w).mean()
        )
        df[f"revenue_rolling_std_{w}"] = (
            df["revenue"].shift(1).rolling(w).std()
        )

    print(f"  Added {len(windows)*2} rolling features")
    return df


def add_category_features(df_raw, df_daily):
    """
    Insight 2 from EDA: category drives revenue patterns.
    Add daily revenue split by category as features.
    """
    cat_daily = (
        df_raw.groupby(["order_date", "category"])["sales"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    cat_daily.columns = (
        ["order_date"] +
        [f"rev_{c.lower().replace(' ','_')}"
         for c in cat_daily.columns[1:]]
    )
    df_daily = df_daily.merge(cat_daily, on="order_date", how="left")
    df_daily[cat_daily.columns[1:]] = (
        df_daily[cat_daily.columns[1:]].fillna(0)
    )
    print(f"  Added {len(cat_daily.columns)-1} category features")
    return df_daily


def add_region_features(df_raw, df_daily):
    """
    Insight 3 from EDA: region performance differs.
    """
    reg_daily = (
        df_raw.groupby(["order_date", "region"])["sales"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    reg_daily.columns = (
        ["order_date"] +
        [f"rev_{c.lower()}" for c in reg_daily.columns[1:]]
    )
    df_daily = df_daily.merge(reg_daily, on="order_date", how="left")
    df_daily[reg_daily.columns[1:]] = (
        df_daily[reg_daily.columns[1:]].fillna(0)
    )
    print(f"  Added {len(reg_daily.columns)-1} region features")
    return df_daily


def split_time_series(df):
    """
    Time series split — NEVER shuffle for time series.
    Future data cannot be used to predict the past.
    Last 20% of dates = test set.
    """
    df = df.dropna().reset_index(drop=True)
    split_idx = int(len(df) * (1 - TEST_SIZE))
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    print(f"\n  Train: {len(train)} days  "
          f"({train['order_date'].min().date()} → "
          f"{train['order_date'].max().date()})")
    print(f"  Test : {len(test)} days   "
          f"({test['order_date'].min().date()} → "
          f"{test['order_date'].max().date()})")
    return train, test


def get_feature_cols(df):
    drop = ["order_date", "revenue", "profit"]
    return [c for c in df.columns if c not in drop]


def save_features(train, test):
    PROC_DIR.mkdir(exist_ok=True)
    train.to_parquet(PROC_DIR / "train_forecast.parquet", index=False)
    test.to_parquet( PROC_DIR / "test_forecast.parquet",  index=False)
    print(f"\n  Saved train + test to {PROC_DIR}")


def run_pipeline():
    print("\n=== Feature Engineering — Forecast Model ===\n")

    df_raw = load_from_warehouse()

    print("\nBuilding daily series...")
    daily = build_daily_series(df_raw)

    print("\nAdding features...")
    daily = add_time_features(daily)
    daily = add_lag_features(daily)
    daily = add_rolling_features(daily)
    daily = add_category_features(df_raw, daily)
    daily = add_region_features(df_raw, daily)

    print(f"\nTotal features: {len(get_feature_cols(daily))}")
    print(f"Total columns : {daily.shape[1]}")

    print("\nSplitting train/test...")
    train, test = split_time_series(daily)

    save_features(train, test)

    print("\n=== Feature engineering complete ===")
    return train, test, get_feature_cols(daily)


if __name__ == "__main__":
    run_pipeline()