import pandas as pd
import numpy as np

seconds_df = pd.read_csv("To_Combine_Seconds_Column.csv")
print(seconds_df)
print(seconds_df.columns)

nan_df = seconds_df.replace(0, np.nan)

print(nan_df.columns)

nan_df['SECONDS'].fillna(nan_df['SECONDS_FROM_MODULO'], inplace=True)
nan_df['SECONDS'].fillna(nan_df['SECONDS_FROM_MINUTES'], inplace=True)
nan_df['SECONDS'].fillna(nan_df['SECONDS_FROM_HOURS'], inplace=True)

print(nan_df)

nan_df.to_csv("Combined_Seconds_Column.csv")
