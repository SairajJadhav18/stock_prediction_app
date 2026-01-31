import yfinance as yf
import os


def download_stock(symbol):
    os.makedirs("data", exist_ok=True)

    df = yf.download(symbol, start="2018-01-01")
    df.reset_index(inplace=True)

    file_path = f"data/{symbol}.csv"
    df.to_csv(file_path, index=False)

    return file_path
