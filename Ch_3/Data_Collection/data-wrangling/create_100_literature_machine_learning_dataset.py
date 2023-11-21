import pandas as pd
import numpy as np


# Pandas and Numpy Options
pd.set_option('display.max_rows', 300, 'display.max_columns', 100)
pd.set_option('use_inf_as_na', True)
np.seterr(divide='ignore', invalid='ignore')

# Import Dataset
our_df = pd.read_excel("Full_Dataset_100_Literature_Paper_Instances_Curated.xls")
our_df_with_duplicates = pd.read_excel("Full_Dataset_100_Literature_Paper_Instances_Curated.xls")
our_df_with_duplicates.set_index("Experiment Number", inplace=True, drop=True)

# add the index of Experiment Number to stop removal of duplicates and re-add paper number
our_df.set_index("Experiment Number", inplace=True, drop=True)
our_df.drop(columns=["Paper Number"], inplace=True)

# Print out the number of duplicates
print(sum(our_df.duplicated()))

# Extract duplicate rows
duplicated_rows = our_df.loc[our_df.duplicated(), :]
our_df.drop_duplicates(inplace=True, keep='first')
print(sum(our_df.duplicated()))
print(our_df.shape)
print(our_df.columns)

# re-add the papers via index
our_df_with_duplicates_papers = our_df_with_duplicates["Paper Number"]
our_df = our_df.join(our_df_with_duplicates_papers)

print(our_df["Paper Number"])

print(our_df.shape)
df_copy_1 = our_df.copy()

# Select Experiments creating a nano structure shape
# print(our_df['Experiment Type'].value_counts())

# Remove all experiments that are imprecise, or do not produce a nano structure shape
df_copy_1 = df_copy_1[df_copy_1['Experiment Type'].isin(['SHAPE'])]

# Remove all Experiments with NaN Magnesium (mM) values
df_copy_1 = df_copy_1[~df_copy_1['Magnesium (mM)'].isin(['NaN', np.NaN])]
print(df_copy_1['Magnesium (mM)'].isnull().sum())

df_copy_1 = df_copy_1[~df_copy_1['Scaffold Name'].isin(['NaN', np.NaN])]

df_copy_1 = df_copy_1[~df_copy_1['Yield Range (%)'].isin(['0-25'])]

# Remove papers that were imprecise
df_copy_1 = df_copy_1[~df_copy_1['Paper Number'].isin(['8', '46', '48', '59', '71', '85', '96', '98'])]

# # Drop irrelevant columns
df_copy_1 = df_copy_1.drop(columns=['General Shape (my interpretation)',
                                    'Actual Shape Name (paper description)',
                                    'Experiment Type', 'Stepwise Detail',
                                    'height (nm)', 'Scaffold to Staple Ratio (Detailed)', 'Appl Yield (%)',
                                    'Thermo-cycler', 'Buffer Name', 'Scaffold Name', 'Scaffold Sequence',
                                    'Boric Acid (mM)', 'Acetic acid (mM)', 'Acetate (mM)', 'Magnesium Acetate Used',
                                    'Unnamed: 0'],
                           errors='ignore')

df_copy_1[['TRIS-HCl (mM)', 'NaCl (mM)',
           'EDTA (mM)']] \
    = df_copy_1[['TRIS-HCl (mM)', 'NaCl (mM)',
                 'EDTA (mM)']].fillna(value=0)

print(df_copy_1.head())
print(df_copy_1.shape)
print(df_copy_1.info())

# Save dataset for machine learning experiments
df_copy_1.to_csv('100_literature_ml_data_set.csv', index=False, header=True)

print(df_copy_1.shape)
print(df_copy_1.columns)

