import pandas as pd
import re
import os
cwd = os.getcwd()

'''
Use this file to help organise and fix staples that are erroneous
for example; 
Filtering during creation of RevNano file formats could show the following:
1) Staples >= length of scaffold
2) Modified staples exist in the staple set, leading to too many staples
3) Far too few staples (only modified staples were given)
So this script is to:
Organise the experiments and papers they originate from to note problems with the papers.

The print out should give:
Experiments separated into number of staples (0, 0-50, 50-100, 1000+)
and should provide details of:
1) Experiment Number and Paper
2) Scaffold Length
3) Staple Length (combined)
4) Number of bases un-hybridised on scaffold (bases remaining un-stapled)

This should show if there is too much over/under-lap which would be unreasonable;
for folding a DNA origami structure.
I can then analyse these papers again to fit into the RevNano software to design features.
'''

# Import the data from path
origami_experiments_path = cwd + "/data_set/data_set_of_magnesium_origami_experiments.xlsx"
staple_data_path = cwd + "/data_set/Staple_Set_Complete_Shared_Fixed_Manually.xlsx"

read_staples_df = pd.read_excel(staple_data_path)
origami_experiments = pd.read_excel(origami_experiments_path)
# print(origami_experiments.columns)

# staple counters
nan_counter = 0
zero_to_fifty_staple_counter = 0
fifty_to_hundred_staple_counter = 0
over_thousand_counter = 0

# fill na with 0 so we can list the missing staple origami experiments
origami_experiments = origami_experiments.fillna(0)

for i in range(len(origami_experiments)):
    # Select from df based upon index value
    origami_df = origami_experiments.iloc[i]
    number_of_staples = origami_df['number of individual staples']
    paper_number = origami_df['Paper Number']
    experiment_number = origami_df['Experiment Number']
    # Analyse Scaffold
    scaffold_sequence = origami_df['Scaffold Sequence']
    scaffold_sequence_len = len(scaffold_sequence)

    # # Select staples based upon experiment number
    staple_set = read_staples_df.loc[read_staples_df['experiment number'].isin([experiment_number])]
    staple_set_list = staple_set['staple sequence'].tolist()

    # CHECK THE STAPLE FOR NON-ACTG CHARACTERS
    for staple_number in range(len(staple_set_list)):
        staple = staple_set_list[staple_number]
        non_actg = re.findall(r"[^ACTG ]", staple)
        if non_actg:
            print("/experiment:", experiment_number, "/paper:", paper_number, "wrong_char:", staple_number)

    # # CHECK THE STAPLE CONTENT FOR MODIFIED PAPERS / EXPERIMENTS
    # for staple_number in range(len(staple_set_list)):
    #     staple = staple_set_list[staple_number]
    #     digit_check = re.findall("\d", staple)  # string contains any digits (numbers from 0-9):
    #     space_check = re.findall("\s", staple)  # string contains any white-space:
    #     lower_case_check = re.findall("[a-z]", staple)  # string contains any lower-case characters:
    #     u_score_check = re.findall("\_", staple)  # string contains any under_scores:
    #     hyphen_check = re.findall("\-", staple)  # string contains any hyphens:
    #     bracket_check = re.findall(r'[(){}[\]]+', staple)  # string contains brackets:
    #     asterix_check = re.findall(r'\*', staple)  # string contains brackets:
    #     if digit_check:
    #         print("/experiment:", experiment_number, "/paper:", paper_number, "digit/row:", staple_number)
    #     if space_check:
    #         print("/experiment:", experiment_number, "/paper:", paper_number, "space/row:", staple_number)
    #     if lower_case_check:
    #         print("/experiment:", experiment_number, "/paper:", paper_number, "low-case/row:", staple_number)
    #     if u_score_check:
    #         print("/experiment:", experiment_number, "/paper:", paper_number, "underscore/row:", staple_number)
    #     if hyphen_check:
    #         print("/experiment:", experiment_number, "/paper:", paper_number, "hyphen/row:", staple_number)
    #     if bracket_check:
    #         print("/experiment:", experiment_number, "/paper:", paper_number, "bracket/row:", staple_number)
    #     if asterix_check:
    #         print("/experiment:", experiment_number, "/paper:", paper_number, "asterix/row:", staple_number)
