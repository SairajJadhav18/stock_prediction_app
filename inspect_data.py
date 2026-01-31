import pandas as pd

df = pd.read_csv("msft_data.csv")

print(df.head())
print(df.columns)

print("\nColumns:", df.columns)
print("\nNumber of rows:", len(df))
