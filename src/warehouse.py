import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DB_URL, PROC_DIR,
    FACT_TABLE, DIM_PRODUCT, DIM_CUSTOMER,
    DIM_LOCATION, DIM_DATE
)


def get_engine():
    return create_engine(DB_URL)


def drop_tables(engine):
    """Drop in correct order — fact first, then dims"""
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {FACT_TABLE}"))
        conn.execute(text(f"DROP TABLE IF EXISTS {DIM_PRODUCT}"))
        conn.execute(text(f"DROP TABLE IF EXISTS {DIM_CUSTOMER}"))
        conn.execute(text(f"DROP TABLE IF EXISTS {DIM_LOCATION}"))
        conn.execute(text(f"DROP TABLE IF EXISTS {DIM_DATE}"))
    print("Existing tables dropped.")


def build_dim_product(df, engine):
    dim = (
        df[["product_id", "product_name",
            "category", "sub_category"]]
        .drop_duplicates(subset=["product_id"])
        .reset_index(drop=True)
    )
    dim.to_sql(DIM_PRODUCT, engine,
               if_exists="replace", index=False)
    print(f"  dim_product    : {len(dim):,} rows")
    return dim


def build_dim_customer(df, engine):
    dim = (
        df[["customer_id", "customer_name", "segment"]]
        .drop_duplicates(subset=["customer_id"])
        .reset_index(drop=True)
    )
    dim.to_sql(DIM_CUSTOMER, engine,
               if_exists="replace", index=False)
    print(f"  dim_customer   : {len(dim):,} rows")
    return dim


def build_dim_location(df, engine):
    dim = (
        df[["city", "state", "region", "country", "postal_code"]]
        .drop_duplicates(subset=["city", "state"])
        .reset_index(drop=True)
    )
    dim["location_id"] = dim.index + 1
    dim.to_sql(DIM_LOCATION, engine,
               if_exists="replace", index=False)
    print(f"  dim_location   : {len(dim):,} rows")
    return dim


def build_dim_date(df, engine):
    dates = pd.DataFrame({
        "date": pd.date_range(
            df["order_date"].min(),
            df["order_date"].max(),
            freq="D"
        )
    })
    dates["year"]        = dates["date"].dt.year
    dates["month"]       = dates["date"].dt.month
    dates["quarter"]     = dates["date"].dt.quarter
    dates["week"]        = dates["date"].dt.isocalendar().week
    dates["day_of_week"] = dates["date"].dt.dayofweek
    dates["month_name"]  = dates["date"].dt.strftime("%B")
    dates["is_weekend"]  = dates["day_of_week"].isin([5, 6]).astype(int)

    dates.to_sql(DIM_DATE, engine,
                 if_exists="replace", index=False)
    print(f"  dim_date       : {len(dates):,} rows")
    return dates


def build_fact_sales(df, dim_location, engine):
    # Join location_id back onto main df
    fact = df.merge(
        dim_location[["city", "state", "location_id"]],
        on=["city", "state"],
        how="left"
    )

    fact = fact[[
        "row_id", "order_id", "order_date", "ship_date",
        "ship_mode", "ship_days",
        "customer_id", "product_id", "location_id",
        "sales", "quantity", "discount", "profit",
        "profit_margin", "discount_impact",
        "order_year", "order_month", "order_quarter"
    ]].copy()

    fact.to_sql(FACT_TABLE, engine,
                if_exists="replace", index=False,
                method="multi", chunksize=500)
    print(f"  fact_sales     : {len(fact):,} rows")
    return fact


def add_indexes(engine):
    """Indexes speed up Tableau and SQL queries dramatically"""
    indexes = [
        f"CREATE INDEX IF NOT EXISTS idx_fact_order_date "
        f"ON {FACT_TABLE}(order_date)",
        f"CREATE INDEX IF NOT EXISTS idx_fact_customer "
        f"ON {FACT_TABLE}(customer_id)",
        f"CREATE INDEX IF NOT EXISTS idx_fact_product "
        f"ON {FACT_TABLE}(product_id)",
        f"CREATE INDEX IF NOT EXISTS idx_fact_location "
        f"ON {FACT_TABLE}(location_id)",
        f"CREATE INDEX IF NOT EXISTS idx_fact_year "
        f"ON {FACT_TABLE}(order_year)",
    ]
    with engine.begin() as conn:
        for idx in indexes:
            conn.execute(text(idx))
    print(f"  Indexes created: {len(indexes)}")


def verify_warehouse(engine):
    queries = {
        "fact_sales":    f"SELECT COUNT(*) FROM {FACT_TABLE}",
        "dim_product":   f"SELECT COUNT(*) FROM {DIM_PRODUCT}",
        "dim_customer":  f"SELECT COUNT(*) FROM {DIM_CUSTOMER}",
        "dim_location":  f"SELECT COUNT(*) FROM {DIM_LOCATION}",
        "dim_date":      f"SELECT COUNT(*) FROM {DIM_DATE}",
    }
    print("\n--- Warehouse row counts ---")
    with engine.connect() as conn:
        for table, query in queries.items():
            count = conn.execute(text(query)).scalar()
            print(f"  {table:<20}: {count:,}")
    print("----------------------------")


def run_warehouse():
    print("\n=== Building Star Schema Warehouse ===\n")

    df = pd.read_parquet(PROC_DIR / "superstore_clean.parquet")
    engine = get_engine()

    drop_tables(engine)

    print("Building dimension tables...")
    dim_product  = build_dim_product(df, engine)
    dim_customer = build_dim_customer(df, engine)
    dim_location = build_dim_location(df, engine)
    dim_date     = build_dim_date(df, engine)

    print("\nBuilding fact table...")
    fact = build_fact_sales(df, dim_location, engine)

    print("\nAdding indexes...")
    add_indexes(engine)

    verify_warehouse(engine)

    print("\n=== Warehouse complete ===")
    engine.dispose()
    return fact


if __name__ == "__main__":
    run_warehouse()