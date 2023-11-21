import pandas as pd
import numpy as np
from os import path

# access the file of interest for reading
basepath = path.dirname(__file__)
filepath = path.abspath(path.join(basepath, "..", "1000_papers_collection_attempt", "master_all_literature_paper_detail.xls"))
print(filepath)

# read the file and select only columns useful for discerning single nano-structures
df = pd.read_excel(filepath)
print(df.columns)

# shape: make sure it is a single, useful structure; experiment number: ID of object in paper; paper number: DOI linked
# Through these three identifiers we know that we are operating on unique structures from a single paper.
headers = ['Experiment Number', 'Paper Number']
useful_df = df[['Experiment Number', 'Paper Number']].copy()
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
    # check = str(row['Shape'])
    # search_string = "Shape"
    # if search_string in check:

    # staple file names created, append to staple sequence list
    staple_seq_filename = ("staple_sequence_paper_" + str(row['Paper Number'])
                           + "_experiment_" + str(int(row['Experiment Number'])))
    unmodified_staple_seq_filename = ("staple_sequence_paper_" + str(row['Paper Number'])
                                      + "_experiment_" + str(int(row['Experiment Number'])) + "_unmodified")
    staple_sequence_files_to_create.append(staple_seq_filename)
    staple_sequence_files_to_create.append(unmodified_staple_seq_filename)

    # scaffold file names created, append to scaffold sequence list
    scaffold_seq_filename = ("scaffold_sequence_paper_" + str(row['Paper Number'])
                             + "_experiment_" + str(int(row['Experiment Number'])))
    unmodified_scaffold_seq_filename = ("scaffold_sequence_paper_" + str(row['Paper Number'])
                                        + "_experiment_" + str(int(row['Experiment Number'])) + "_unmodified")
    scaffold_sequence_files_to_create.append(scaffold_seq_filename)
    scaffold_sequence_files_to_create.append(unmodified_scaffold_seq_filename)

for filename in scaffold_sequence_files_to_create:
    save_path = "scaffold_sequences/"
    completeName = path.join(save_path + filename + ".txt")
    file1 = open(completeName, "w")
    file1.close()

for filename in staple_sequence_files_to_create:
    save_path = "staple_sequences/"
    completeName = path.join(save_path + filename + ".txt")
    file1 = open(completeName, "w")
    file1.close()

