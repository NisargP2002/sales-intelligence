import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    RAW_DIR, PROC_DIR, RAW_FILE, DB_URL
)


def extract(filename=RAW_FILE):
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found at {path}. Did you download it?"
        )
    df = pd.read_csv(path, encoding="latin-1")
    print(f"Extracted: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def transform(df):
    df = df.copy()

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.lower()
    )

    # Parse dates
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["ship_date"]  = pd.to_datetime(df["ship_date"])

    # Derived date columns
    df["order_year"]    = df["order_date"].dt.year
    df["order_month"]   = df["order_date"].dt.month
    df["order_quarter"] = df["order_date"].dt.quarter
    df["order_dow"]     = df["order_date"].dt.dayofweek
    df["ship_days"]     = (
        df["ship_date"] - df["order_date"]
    ).dt.days

    # Fix numeric columns
    for col in ["sales", "profit", "discount", "quantity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived business metrics
    df["profit_margin"] = np.where(
        df["sales"] > 0,
        (df["profit"] / df["sales"]).round(4),
        0
    )
    df["discount_impact"] = (df["discount"] * df["sales"]).round(2)

    # Drop nulls on critical columns
    before = len(df)
    df = df.dropna(subset=["sales", "order_date"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with null sales/dates")

    print(f"Transformed: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate(df):
    checks = {
        "No negative sales":      (df["sales"] >= 0).all(),
        "No null order dates":    df["order_date"].notna().all(),
        "No null regions":        df["region"].notna().all(),
        "Sales > 0":              (df["sales"] > 0).sum() > 0,
       "Profit margin in range": df["profit_margin"].between(-3, 1).all(),
        "Ship days >= 0":         (df["ship_days"] >= 0).all(),
    }
    print("\n--- Validation ---")
    all_passed = True
    for check, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"  {status}  {check}")
        if not result:
            all_passed = False
    print("------------------")
    if not all_passed:
        raise ValueError("Validation failed.")
    return df


def save_parquet(df):
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = PROC_DIR / "superstore_clean.parquet"
    df.to_parquet(out, index=False)
    print(f"Parquet saved → {out}  "
          f"({out.stat().st_size / 1024:.1f} KB)")


def load_to_postgres(df):
    """
    Load cleaned flat table into PostgreSQL.
    warehouse.py will split this into star schema tables.
    """
    engine = create_engine(DB_URL)

    # Test connection first
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("PostgreSQL connection OK")

    # Write raw cleaned table
    df.to_sql(
        "stg_superstore",
        engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=500
    )
    print(f"Loaded {len(df)} rows → stg_superstore")
    engine.dispose()


def run_etl():
    print("\n=== ETL Pipeline ===\n")
    df = extract()
    df = transform(df)
    df = validate(df)
    save_parquet(df)
    load_to_postgres(df)

    print("\n=== ETL complete ===")
    print(f"  Date range  : {df['order_date'].min().date()}"
          f" → {df['order_date'].max().date()}")
    print(f"  Revenue     : ${df['sales'].sum():,.0f}")
    print(f"  Orders      : {df['order_id'].nunique():,}")
    print(f"  Customers   : {df['customer_id'].nunique():,}")
    print(f"  Products    : {df['product_id'].nunique():,}")
    print(f"  Regions     : {df['region'].unique().tolist()}")
    return df


if __name__ == "__main__":
    run_etl()