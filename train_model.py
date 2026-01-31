import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from download_data import download_stock
from feature_engineering import engineer_features


def run_pipeline(symbol):
    os.makedirs("outputs", exist_ok=True)

    download_stock(symbol)
    df = engineer_features(symbol)
    

    features = [
    "return_1",
    "return_2",
    "return_5",
    "ma_5",
    "ma_10",
    "volatility_5",
    "day_of_week"
]
    print("Available columns:", df.columns.tolist())
    print("Requested features:", features)


    X = df[features]
    y = df["return"].shift(-1)

    X = X.iloc[:-1]
    y = y.iloc[:-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    directional_accuracy = np.mean(
        np.sign(y_test.values) == np.sign(y_pred)
    )

    pd.DataFrame([{
        "mae": mae,
        "r2": r2,
        "directional_accuracy": directional_accuracy
    }]).to_json(
        f"outputs/metrics_{symbol}.json",
        orient="records",
        indent=2
    )

    pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values(
        "importance", ascending=False
    ).to_json(
        f"outputs/feature_importance_{symbol}.json",
        orient="records",
        indent=2
    )

    pd.DataFrame([{
        "company": symbol,
        "symbol": symbol,
        "last_close": float(df["Close"].iloc[-1]),
        "last_date": str(df["Date"].iloc[-1].date())
    }]).to_json(
        f"outputs/stock_info_{symbol}.json",
        orient="records",
        indent=2
    )
