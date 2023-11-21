import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from textwrap import wrap

# Directories containing files
cwd = os.getcwd()

# change the path to the correct file path
path = "C:/Users/Big JC/PycharmProjects/PhDCodeOnly/ML_results_to_analyse/analyse_16_11_2020/5CV_Coarse_RFE/" \
       "5CV Coarse/extra_trees_results/model_final_scores/" \
       "Extra_Trees_RFE_5CV_Stratified_Baseline_final_scores.csv"
columns = ["r2", "MAE", "RMSE", "MSE", "MedianAE"]

df = pd.read_csv(path, names=columns)
df.index += 1

df.sort_values(by=["r2"], inplace=True)
df_low = pd.DataFrame(data=df.iloc[0]).transpose()
df_high = pd.DataFrame(data=df.iloc[-1]).transpose()
df_average = pd.DataFrame(data=df.apply(np.average), columns=["Average"]).transpose()
df_stdev = pd.DataFrame(data=df.apply(np.std), columns=["StDev"]).transpose()

total_result_df = df_low, df_high, df_average, df_stdev
result = pd.concat(total_result_df)
# creates a table for best and worst repetition
print(result)

