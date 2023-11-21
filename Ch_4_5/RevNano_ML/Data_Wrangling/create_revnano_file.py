import pandas as pd
import numpy as np
import csv
import glob
import os
cwd = os.getcwd()

'''
RevNano Requirements include: 
Scaffold (longest sequence) must be at the top of the file,
Staples for that file must follow.
- Structures available -
Planar Structures, Un-modified Strand Structures,
Structures that aren't very large (Lambda is an example of a large structure >46Kb)

This script will aim to: 
1) ID: Identify the Experiment Number: I.E: 16.
2) - Additional - Filter the Experiments based on sequence length
3) OPEN the file that contains Scaffold (E.G: Data Set File V3).
4) APPEND the Scaffold to the first line of a CSV.
5) OPEN the file that contains the origami staples.
6) APPEND the Origami staples to the rest of the CSV lines.
7) - Additional - Remove obviously modified staples.
8) Once all files are created, this script is complete, next stage: Call RevNano
'''


# Import the data from path
scaffold_data_path = cwd + "/data_set/Full_Dataset_V3_Shared.xlsx"
staple_data_path = cwd + "/data_set/Staple_Set_Complete_Shared_Fixed_Manually.xlsx"

# Read the data from path
read_scaffold_df = pd.read_excel(scaffold_data_path)
read_staples_df = pd.read_excel(staple_data_path)

# Filter the data set
scaffold_exists_df = read_scaffold_df[~read_scaffold_df['Scaffold Sequence'].isin([np.NaN])]
shape_exists_df = scaffold_exists_df[scaffold_exists_df['Experiment Type'].isin(['SHAPE'])]
magnesium_nan_df = shape_exists_df[~shape_exists_df['Magnesium (mM)'].isin(['NaN'])]
yield_filter_df = magnesium_nan_df[~magnesium_nan_df['Yield Range (%)'].isin(['0-25'])]

print(yield_filter_df.shape)
length_of_df = int(yield_filter_df.shape[0])
scaffold_df = yield_filter_df.reset_index(drop=True)
total_test_origami = []
counter = 0
counter_available = 0
counter_incompatible = 0

# sanity check lists
check_hybrid_list = []
few_staples_experiments_list = []
small_scaffold_experiments_list = []
small_scaffold_len = []

# check papers
paper_list = []

# loop over data
for origami in range(0, length_of_df):
    # # Sanity Checker
    # print(origami)
    # print(counter)
    counter += 1

    # test origami
    test_origami = []

    # Select scaffold based upon index value
    selected_df = scaffold_df.iloc[origami]
    # print("Paper Number: ", selected_df['Paper Number'])
    experiment_number = selected_df['Experiment Number']
    scaffold_sequence = selected_df['Scaffold Sequence']
    paper_number = selected_df['Paper Number']
    scaffold_sequence_length = len(scaffold_sequence)
    # print("Scaffold Name: ", selected_df['Scaffold Name'])

    # # Select staples based upon experiment number
    staple_set = read_staples_df.loc[read_staples_df['experiment number'].isin([experiment_number])]
    staple_set_list = staple_set['staple sequence'].tolist()

    # Sanity Checks #################
    if scaffold_sequence_length <= 5000:
        small_scaffold_len.append(scaffold_sequence_length)
        small_scaffold_experiments_list.append(experiment_number)

    # # Append to the Origami List
    scaffold_sequence = scaffold_sequence.rstrip("\n")
    test_origami.append(scaffold_sequence)
    total_seq_size = "".join(staple_set_list)
    total_staple_seq_size = len(total_seq_size)
    check_hybridisation = (scaffold_sequence_length - total_staple_seq_size)

    if check_hybridisation <= 100:
        check_hybrid_list.append(check_hybridisation)
        few_staples_experiments_list.append(experiment_number)
    ########################################

    for staple in staple_set_list:
        test_origami.append(staple)

    if total_staple_seq_size != 0:
        if check_hybridisation >= 0:
            counter_available += 1
            paper_list.append(paper_number)
            print("Paper:", paper_number, "/Experiment Number:", experiment_number, "/scaffold:",
                  scaffold_sequence_length,
                  "/staples bases:", total_staple_seq_size, "/un-hybridised:", check_hybridisation)
            # Write to CSV
            filename = "origami_experiment_" + str(experiment_number) + "_sequences.rev"
            with open("run_through/" + filename, "w") as outfile:
                for seq in test_origami:
                    outfile.write(seq)
                    outfile.write("\n")

    if total_staple_seq_size > 10:
        if check_hybridisation <= -1:
            counter_incompatible += 1
            paper_list.append(paper_number)
            print("Paper:", paper_number, "/Experiment Number:", experiment_number, "/scaffold:",
                  scaffold_sequence_length,
              "/staples bases:", total_staple_seq_size, "/un-hybridised:", check_hybridisation)
            # Write to CSV
            filename = "origami_experiment_" + str(experiment_number) + "_sequences.rev"
            with open("check_scaffold/" + filename, "w") as outfile:
                for seq in test_origami:
                    outfile.write(seq)
                    outfile.write("\n")

    filename = "origami_experiment_" + str(experiment_number) + "_sequences.rev"
    with open(filename, "w") as outfile:
        for seq in test_origami:
            outfile.write(seq)
            outfile.write("\n")

too_many_staples = list(zip(check_hybrid_list, few_staples_experiments_list))
too_many_staples_df = pd.DataFrame(too_many_staples)
print(too_many_staples_df.shape)
# too_many_staples_df.to_csv("too_many_staples_list.csv")

small_scaffolds = list(zip(small_scaffold_len, small_scaffold_experiments_list))
small_scaffold_df = pd.DataFrame(small_scaffolds)
print(small_scaffold_df)
# small_scaffold_df.to_csv("small_scaffold_list.csv")

print(counter_available)
print(counter_incompatible)
print(paper_list)
