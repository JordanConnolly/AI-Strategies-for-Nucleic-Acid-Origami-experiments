import pandas as pd
import numpy as np
from os import path

# access the file of interest for reading
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "staple_descriptors",
                                  "Staple_Set_Complete_Shared_Fixed_Manually_100_Initial_Papers.xls"))
print(filepath)
# read in the excel spread-sheet, where there are staples for the initial 100 papers
# read the file and select only columns useful for discerning single nano-structures
df = pd.read_excel(filepath)
print(df.columns)

# shape: make sure it is a single, useful structure; experiment number: ID of object in paper; paper number: DOI linked
# Through these three identifiers we know that we are operating on unique structures from a single paper.
headers = ['experiment number', 'paper number']
useful_df = df[['experiment number', 'paper number']].copy()
print(useful_df.columns)

# count the useful structures (those that are single shapes)
# counter = 0
# for index, row in useful_df.iterrows():
#     check = str(row['Shape'])
#     search_string = "Shape"
#     if search_string in check:
#         print(row['Shape'], row['Experiment Number'], row['Paper Number'])
#         counter += 1
# print("total count of useful single structures:", counter)

# discover the number of duplicates (in case human error / ID system failed)
duplicated_df = useful_df.duplicated()
not_duplicate_counter = 0
for row in duplicated_df.values:
    if row is True:
        print("Duplicate Row")
    else:
        not_duplicate_counter += 1
print("number of unique rows", not_duplicate_counter)
print("total number of rows", len(duplicated_df.values))
print("number of duplicates", not_duplicate_counter - len(duplicated_df.values))

# With checks complete, create file names to store sequences
staple_sequence_files_to_create = []
scaffold_sequence_files_to_create = []
for index, row in df.iterrows():
    unmodified_staple_seq_filename = ("staple_sequence_paper_" + str(row['paper number'])
                                      + "_experiment_" + str(int(row['experiment number'])) + "_unmodified")
    staple_sequence_files_to_create.append(unmodified_staple_seq_filename)

for filename in set(staple_sequence_files_to_create):
    paper = filename.split("_")[3]
    experiment = filename.split("_")[5]
    print(paper, experiment)
    correct_paper_check = df.loc[df['paper number'] == int(paper)]
    correct_experiment_check = correct_paper_check.loc[correct_paper_check['experiment number'] == int(experiment)]
    staple_sequences_for_experiment = correct_experiment_check['staple sequence'].values
    storage_path = "staple_descriptors/staple_sequences/"
    completeName = path.join(storage_path + filename + ".txt")
    f = open(completeName, "w")
    for i in staple_sequences_for_experiment:
        f.write(f'{i}\n')
    f.close()
