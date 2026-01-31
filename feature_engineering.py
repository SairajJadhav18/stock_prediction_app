import pandas as pd


def engineer_features(symbol):
    df = pd.read_csv(f"data/{symbol}.csv", index_col=0)

    df.index.name = "Date"
    df.reset_index(inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["return"] = df["Close"].pct_change()

    df["return_1"] = df["return"].shift(1)
    df["return_2"] = df["return"].shift(2)
    df["return_5"] = df["return"].shift(5)

    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()

    df["volatility_5"] = df["return"].rolling(5).std()

    df["day_of_week"] = df["Date"].dt.dayofweek

    df["target"] = df["return"].shift(-1)

    df.dropna(inplace=True)

    return df
