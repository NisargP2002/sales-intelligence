import sys
import joblib
import numpy as np
import pandas as pd
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    MODEL_DIR, PROC_DIR, DB_URL,
    API_HOST, API_PORT
)
from api.schema import (
    ForecastRequest, ForecastResponse, ForecastPoint,
    SummaryResponse, RegionRevenueItem, CategoryRevenueItem
)

model        = None
feature_cols = None
all_data     = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_cols, all_data

    print("Loading forecast model...")
    model = joblib.load(MODEL_DIR / "forecast_model.pkl")

    train = pd.read_parquet(PROC_DIR / "train_forecast.parquet")
    test  = pd.read_parquet(PROC_DIR / "test_forecast.parquet")
    all_data = pd.concat([train, test]).reset_index(drop=True)

    drop  = ["order_date", "revenue", "profit"]
    feature_cols = [c for c in train.columns if c not in drop]

    print(f"Model ready — {len(feature_cols)} features")
    yield
    model = None


app = FastAPI(
    title="Sales Intelligence API",
    description="Revenue forecasting powered by XGBoost",
    version="1.0.0",
    lifespan=lifespan
)

engine = create_engine(DB_URL)


# ── helpers ──────────────────────────────────────────────────

def predict_next_n_days(days: int) -> pd.DataFrame:
    last_date = all_data["order_date"].max()
    future    = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=days, freq="D"
    )
    history = all_data["revenue"].tolist()
    preds   = []

    for d in future:
        row = {
            "year":           d.year,
            "month":          d.month,
            "quarter":        d.quarter,
            "day_of_week":    d.dayofweek,
            "day_of_year":    d.dayofyear,
            "week":           d.isocalendar()[1],
            "is_weekend":     int(d.dayofweek in [5, 6]),
            "is_month_end":   int(d.is_month_end),
            "is_month_start": int(d.is_month_start),
            "is_q4":          int(d.quarter == 4),
            "revenue_lag_1":  history[-1],
            "revenue_lag_7":  history[-7]  if len(history) >= 7  else 0,
            "revenue_lag_14": history[-14] if len(history) >= 14 else 0,
            "revenue_lag_30": history[-30] if len(history) >= 30 else 0,
        }
        for w in [7, 14, 30]:
            window = history[-w:] if len(history) >= w else history
            row[f"revenue_rolling_mean_{w}"] = np.mean(window)
            row[f"revenue_rolling_std_{w}"]  = np.std(window)

        for col in feature_cols:
            if col not in row:
                row[col] = float(
                    all_data[col].iloc[-30:].mean()
                    if col in all_data.columns else 0
                )

        X    = pd.DataFrame([row])[feature_cols]
        pred = max(float(model.predict(X)[0]), 0)
        preds.append(pred)
        history.append(pred)

    return pd.DataFrame({"date": future, "forecast": preds})


# ── routes ───────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded")
    try:
        df = predict_next_n_days(req.days)
        return ForecastResponse(
            period_start=str(df["date"].iloc[0].date()),
            period_end=str(df["date"].iloc[-1].date()),
            total_forecast=round(df["forecast"].sum(), 2),
            daily_avg=round(df["forecast"].mean(), 2),
            points=[
                ForecastPoint(
                    date=str(r["date"].date()),
                    forecast=round(r["forecast"], 2)
                )
                for _, r in df.iterrows()
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary", response_model=SummaryResponse)
def summary():
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT
                    ROUND(SUM(f.sales)::numeric, 2)  AS total_revenue,
                    ROUND(SUM(f.profit)::numeric, 2) AS total_profit,
                    COUNT(DISTINCT f.order_id)        AS total_orders
                FROM fact_sales f
            """)).fetchone()

            best_region = conn.execute(text("""
                SELECT l.region
                FROM fact_sales f
                JOIN dim_location l ON f.location_id = l.location_id
                GROUP BY l.region
                ORDER BY SUM(f.sales) DESC
                LIMIT 1
            """)).scalar()

            best_cat = conn.execute(text("""
                SELECT p.category
                FROM fact_sales f
                JOIN dim_product p ON f.product_id = p.product_id
                GROUP BY p.category
                ORDER BY SUM(f.sales) DESC
                LIMIT 1
            """)).scalar()

            dates = conn.execute(text("""
                SELECT MIN(order_date), MAX(order_date)
                FROM fact_sales
            """)).fetchone()

        total_days = (dates[1] - dates[0]).days or 1
        return SummaryResponse(
            total_revenue=float(row[0]),
            total_profit=float(row[1]),
            total_orders=int(row[2]),
            avg_daily_revenue=round(float(row[0]) / total_days, 2),
            best_region=best_region,
            best_category=best_cat,
            date_range=f"{dates[0].date()} → {dates[1].date()}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/revenue/by-region",
         response_model=list[RegionRevenueItem])
def revenue_by_region():
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    l.region,
                    ROUND(SUM(f.sales)::numeric, 2)  AS revenue,
                    ROUND(SUM(f.profit)::numeric, 2) AS profit
                FROM fact_sales f
                JOIN dim_location l ON f.location_id = l.location_id
                GROUP BY l.region
                ORDER BY revenue DESC
            """)).fetchall()
        return [
            RegionRevenueItem(
                region=r[0],
                revenue=float(r[1]),
                profit=float(r[2])
            ) for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/revenue/by-category",
         response_model=list[CategoryRevenueItem])
def revenue_by_category():
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    p.category,
                    ROUND(SUM(f.sales)::numeric, 2) AS revenue,
                    ROUND(AVG(f.profit_margin)::numeric * 100, 2)
                        AS margin
                FROM fact_sales f
                JOIN dim_product p ON f.product_id = p.product_id
                GROUP BY p.category
                ORDER BY revenue DESC
            """)).fetchall()
        return [
            CategoryRevenueItem(
                category=r[0],
                revenue=float(r[1]),
                margin=float(r[2])
            ) for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api.main:app",
                host=API_HOST,
                port=API_PORT,
                reload=True)