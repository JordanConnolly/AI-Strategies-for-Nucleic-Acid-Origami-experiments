import pandas as pd
import numpy as np

seconds_df = pd.read_csv("To_Combine_Seconds_Curve_3.csv")
print(seconds_df)
print(seconds_df.columns)

all_lit_df = pd.read_excel("All_Literature_Papers_Temperature_Ramp_V2.xls")
print(all_lit_df)

# add the column where curve is correct value
merged_curves_df = pd.merge(left=all_lit_df, right=seconds_df, how='outer', on='CURVE')
print(merged_curves_df)

# combine the seconds columns to create one column
merged_curves_df['TOTAL_SECONDS'].fillna(merged_curves_df['SECONDS COMBINED'], inplace=True)
print(merged_curves_df)

# aggregate the values
aggregation_functions = {'TOTAL_SECONDS': 'sum'}
final_seconds_df = merged_curves_df.groupby(merged_curves_df['Experiment Number']).aggregate(aggregation_functions)
print(final_seconds_df)

# combine into a final data frame
all_lit_final_df = pd.read_excel("all_instances_literature_dataset.xls")
merged_lit_df = pd.merge(left=all_lit_final_df, right=final_seconds_df, how='outer', on='Experiment Number')
# print(merged_lit_df)
merged_lit_df['TOTAL_SECONDS'] = merged_lit_df['TOTAL_SECONDS'].replace(0, np.nan)

print(merged_lit_df)
# merged_lit_df.to_excel("all_instances_literature_dataset_seconds_fixed_2.xls")
# print(merged_lit_df.columns)

