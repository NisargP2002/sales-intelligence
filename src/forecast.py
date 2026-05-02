import sys
import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PROC_DIR, MODEL_DIR, OUTPUT_DIR,
    N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE,
    RANDOM_STATE
)


def load_data():
    train = pd.read_parquet(PROC_DIR / "train_forecast.parquet")
    test  = pd.read_parquet(PROC_DIR / "test_forecast.parquet")
    print(f"Train: {train.shape}  |  Test: {test.shape}")
    return train, test


def get_feature_cols(df):
    drop = ["order_date", "revenue", "profit"]
    return [c for c in df.columns if c not in drop]


def train(train_df, test_df, run_name="xgb_forecast_v1"):
    MODEL_DIR.mkdir(exist_ok=True)

    feature_cols = get_feature_cols(train_df)
    X_train = train_df[feature_cols]
    y_train = train_df["revenue"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["revenue"]

    mlflow.set_experiment("sales_forecast")

    with mlflow.start_run(run_name=run_name) as run:
        params = {
            "n_estimators":  N_ESTIMATORS,
            "max_depth":     MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "random_state":  RANDOM_STATE,
            "subsample":     0.8,
            "colsample_bytree": 0.8,
            "tree_method":   "hist"
        }
        mlflow.log_params(params)

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50
        )

        # Save model
        model_path = MODEL_DIR / "forecast_model.pkl"
        joblib.dump(model, model_path)
        mlflow.xgboost.log_model(model, "model")

        print(f"\nModel saved → {model_path}")
        print(f"MLflow run  → {run.info.run_id}")
        return model, feature_cols


def evaluate(model, test_df, feature_cols):
    X_test = test_df[feature_cols]
    y_test = test_df["revenue"]

    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # revenue can't be negative

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(
        np.abs((y_test - y_pred) / np.where(y_test > 0, y_test, 1))
    ) * 100

    print("\n=== Forecast Evaluation ===")
    print(f"  MAE  : ${mae:,.0f}   (avg daily error)")
    print(f"  RMSE : ${rmse:,.0f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.1f}%")

    mlflow.set_experiment("sales_forecast")
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({
            "mae": mae, "rmse": rmse,
            "r2": r2, "mape": mape
        })

    return y_pred, {"mae": mae, "rmse": rmse,
                    "r2": r2, "mape": mape}


def plot_results(test_df, y_pred, metrics):
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    dates  = test_df["order_date"]
    y_test = test_df["revenue"]

    # ── Plot 1: Actual vs predicted ──────────────────────
    axes[0,0].plot(dates, y_test, color="#534AB7",
                   label="Actual", linewidth=1.5)
    axes[0,0].plot(dates, y_pred, color="#D85A30",
                   label="Predicted", linewidth=1.5,
                   linestyle="--")
    axes[0,0].set_title("Actual vs Predicted revenue")
    axes[0,0].set_xlabel("Date")
    axes[0,0].set_ylabel("Daily Revenue ($)")
    axes[0,0].legend()

    # ── Plot 2: Residuals ────────────────────────────────
    residuals = y_test.values - y_pred
    axes[0,1].scatter(y_pred, residuals,
                      alpha=0.4, color="#534AB7", s=10)
    axes[0,1].axhline(0, color="#D85A30",
                      linestyle="--", linewidth=1)
    axes[0,1].set_xlabel("Predicted")
    axes[0,1].set_ylabel("Residual")
    axes[0,1].set_title("Residual plot")

    # ── Plot 3: Feature importance ───────────────────────
    feat_names = test_df[
        [c for c in test_df.columns
         if c not in ["order_date","revenue","profit"]]
    ].columns
    importances = model.feature_importances_
    idx = np.argsort(importances)[-15:]
    axes[1,0].barh(feat_names[idx],
                   importances[idx],
                   color="#1D9E75", alpha=0.8)
    axes[1,0].set_title("Top 15 feature importances")
    axes[1,0].set_xlabel("Importance")

    # ── Plot 4: Prediction error distribution ────────────
    pct_errors = np.abs(residuals / np.where(
        y_test > 0, y_test, 1)) * 100
    axes[1,1].hist(pct_errors, bins=40,
                   color="#534AB7", alpha=0.7)
    axes[1,1].axvline(metrics["mape"], color="#D85A30",
                      linestyle="--",
                      label=f"MAPE = {metrics['mape']:.1f}%")
    axes[1,1].set_xlabel("Absolute % error")
    axes[1,1].set_ylabel("Count")
    axes[1,1].set_title("Error distribution")
    axes[1,1].legend()

    plt.suptitle("Sales Forecast Model Evaluation",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / "forecast_evaluation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")


def future_forecast(model, train_df, test_df,
                    feature_cols, days=90):
    """
    Forecast 90 days beyond the last known date.
    Uses recursive strategy: each prediction becomes
    a lag feature for the next prediction.
    """
    all_data = pd.concat([train_df, test_df]).reset_index(drop=True)
    last_date = all_data["order_date"].max()

    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=days, freq="D"
    )

    preds = []
    history = all_data["revenue"].tolist()

    for d in future_dates:
        row = {
            "year":          d.year,
            "month":         d.month,
            "quarter":       d.quarter,
            "day_of_week":   d.dayofweek,
            "day_of_year":   d.dayofyear,
            "week":          d.isocalendar()[1],
            "is_weekend":    int(d.dayofweek in [5, 6]),
            "is_month_end":  int(d.is_month_end),
            "is_month_start":int(d.is_month_start),
            "is_q4":         int(d.quarter == 4),
            "revenue_lag_1": history[-1],
            "revenue_lag_7": history[-7]  if len(history)>=7  else 0,
            "revenue_lag_14":history[-14] if len(history)>=14 else 0,
            "revenue_lag_30":history[-30] if len(history)>=30 else 0,
        }

        # Rolling means from history
        for w in [7, 14, 30]:
            window = history[-w:] if len(history) >= w else history
            row[f"revenue_rolling_mean_{w}"] = np.mean(window)
            row[f"revenue_rolling_std_{w}"]  = np.std(window)

        # Fill remaining features with recent averages
        for col in feature_cols:
            if col not in row:
                if col in all_data.columns:
                    row[col] = all_data[col].iloc[-30:].mean()
                else:
                    row[col] = 0

        X = pd.DataFrame([row])[feature_cols]
        pred = float(model.predict(X)[0])
        pred = max(pred, 0)
        preds.append(pred)
        history.append(pred)

    forecast_df = pd.DataFrame({
        "date":     future_dates,
        "forecast": preds
    })

    # Save forecast
    out = PROC_DIR / "future_forecast.parquet"
    forecast_df.to_parquet(out, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    recent = all_data.tail(90)
    ax.plot(recent["order_date"], recent["revenue"],
            color="#534AB7", label="Historical", linewidth=1.5)
    ax.plot(forecast_df["date"], forecast_df["forecast"],
            color="#D85A30", label="90-day forecast",
            linewidth=2, linestyle="--")
    ax.axvline(last_date, color="#888780",
               linestyle=":", linewidth=1)
    ax.set_title("90-day revenue forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Revenue ($)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "future_forecast.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    total = forecast_df["forecast"].sum()
    print(f"\n=== 90-day forecast ===")
    print(f"  Period  : {future_dates[0].date()} → "
          f"{future_dates[-1].date()}")
    print(f"  Total   : ${total:,.0f}")
    print(f"  Daily avg: ${forecast_df['forecast'].mean():,.0f}")
    print(f"  Saved   → {out}")

    return forecast_df


if __name__ == "__main__":
    print("\n=== Sales Forecast Pipeline ===\n")
    train_df, test_df = load_data()
    model, feature_cols = train(train_df, test_df)
    y_pred, metrics = evaluate(model, test_df, feature_cols)
    plot_results(test_df, y_pred, metrics)
    forecast_df = future_forecast(
        model, train_df, test_df, feature_cols)
    print("\n=== Pipeline complete ===")