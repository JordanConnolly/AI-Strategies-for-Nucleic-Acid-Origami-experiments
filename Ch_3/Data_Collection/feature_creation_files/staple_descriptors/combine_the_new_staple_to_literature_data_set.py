import pandas as pd
import re
import glob
import numpy as np

lit_df = pd.read_excel("all_instances_literature_dataset_in_progress.xls")
staple_df = pd.read_csv("new_bolstered_descriptive_staple_file_all_literature.csv")

merged_df = pd.merge(lit_df, staple_df, how='outer', on='Experiment Number')
print(merged_df.columns)
merged_df = merged_df.drop(columns=["Unnamed: 0_x", 'Unnamed: 0.1'])
print(merged_df)
# # Save with no NaN drop

# created with anomalous results investigated manually
# Equivalent to Full Dataset V3, where mixtures will be removed, leaving SHAPES only.
merged_df.to_excel("Full_Dataset_V1_Curated_All_Literature_Instances.xls")
