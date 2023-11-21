import pandas as pd
import numpy as np


# Pandas and Numpy Options
pd.set_option('display.max_rows', 300, 'display.max_columns', 100)
pd.set_option('use_inf_as_na', True)
np.seterr(divide='ignore', invalid='ignore')

# Import Dataset
our_df = pd.read_excel("Full_Dataset_All_Literature_Instances_Curated.xls")

print(our_df.shape)
df_copy_1 = our_df.copy()

# Select Experiments creating a nano structure shape
# print(our_df['Experiment Type'].value_counts())

# Remove all experiments that are imprecise, or do not produce a nano structure shape
# df_copy_1 = df_copy_1[~df_copy_1['Experiment Type'].isin(['MIXTURE',
#                                                           'SHAPE - Imprecise + MD Simulations '
#                                                           '(useful for Structure Simulation)',
#                                                           'SHAPE - Imprecise', 'SHAPE - Identical to another'])]

# Remove all Experiments with NaN Magnesium (mM) values
# df_copy_1 = df_copy_1[~df_copy_1['Magnesium (mM)'].isin(['NaN', np.NaN])]
# print(df_copy_1['Magnesium (mM)'].isnull().sum())

# Remove papers that were imprecise
# df_copy_1 = df_copy_1[~df_copy_1['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]

# # Drop irrelevant columns
# df_copy_1 = df_copy_1.drop(columns=['Scaffold Sequence',
#                                     'General Shape (my interpretation)',
#                                     'Actual Shape Name (paper description)',
#                                     'Experiment Type', 'Unnamed: 0', 'Stepwise Detail',
#                                     'height (nm)', 'Scaffold to Staple Ratio (Detailed)', 'Appl Yield (%)',
#                                     'Thermo-cycler'])

# # Drop irrelevant columns
df_copy_1 = df_copy_1.drop(columns=['Unnamed: 0'])

df_copy_1[['TRIS-HCl (mM)', 'Boric Acid (mM)', 'NaCl (mM)', 'Acetate (mM)',
           'Acetic acid (mM)', 'EDTA (mM)']] \
    = df_copy_1[['TRIS-HCl (mM)', 'Boric Acid (mM)', 'NaCl (mM)', 'Acetate (mM)',
                 'Acetic acid (mM)', 'EDTA (mM)']].fillna(value=0)

print(df_copy_1.head())
print(df_copy_1.shape)
print(df_copy_1.info())

# Save dataset for machine learning experiments
df_copy_1.to_excel('full_dataset_all_literature_instances_curated.xls', index=False, header=True)

