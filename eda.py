import os
import pandas as pd

print(pd.__version__)

df_clicks = pd.read_parquet("./data/clicks.parquet")
df_stores = pd.read_parquet("./data/stores.parquet")
df_users = pd.read_parquet("./data/users.parquet")

print(df_clicks.sample().T)
