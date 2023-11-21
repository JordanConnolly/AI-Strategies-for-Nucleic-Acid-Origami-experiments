import pandas as pd
import numpy as np
import os
from textwrap import wrap
import glob
import re
import matplotlib.pyplot as plt

# print(plt.style.available)
plt.style.use('seaborn-whitegrid')
path_parent = os.path.dirname(os.getcwd())


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def create_list_of_floats(series):
    new_list = []
    for i in series:
        if "e" in i:
            metric_list = i.replace(" ", ",").strip("]").strip("[").split(",")
            new_list.append(metric_list)
        else:
            strip_i = i.strip("[,").strip("[ ").strip("]")
            metric_list = ' '.join(strip_i.split()).replace(" ", ",")
            metric_list = metric_list.split(",")
            new_list.append(metric_list)
    return new_list


# # # all origami list
# list_of_origami_names = ["DBS_square",
#                          "DBS_square",
#                          "DBS_square",
#                          "DBS_square",
#                          "DBS_square",
#                          "DBS_square"]
#
# list_of_origami_names_for_plot = ["Jurek Square Sweep 1",
#                                   "Jurek Square Sweep 2",
#                                   "Jurek Square Sweep 3",
#                                   "Jurek Square Sweep 4",
#                                   "Jurek Square Sweep 5",
#                                   "Jurek Square Sweep 6"
#                                   ]

list_of_origami_names = ["DBS_square"]
list_of_origami_names_for_plot = ["Jurek Square Sweep 5"]

large_origami_list = []
plot_name = "4 Metric DBS Initial Results"
save_name = "4_Metric_DBS_Initial_Results"
limit_num_of_generations = 10

# loop over logbook, get the min fit for all gens, plot as a line
# now we need to loop over origamis, store their logbook as separate groups, plot lines

# store all reps for an experiment
fit_min_floats_all_reps = pd.DataFrame()

# for origami_name in list_of_origami_names:
name_counter = 0
for origami_name in list_of_origami_names:
    sweep_list = ["1", "2", "3", "4", "5", "6"]
    # sweep_list = ["1", "2", "4", "5"]
    # sweep_list = "4"
    current_sweep = sweep_list[4]

    # current_sweep = sweep_list[name_counter]

    experiment_name = origami_name + "_sweep_" + current_sweep + "/"
    origami_name_for_plot = list_of_origami_names_for_plot[name_counter]

    experiment_list = []

    if current_sweep == "1" or "2" or "3" or "4" or "6":
        experiment_list = [
            "DBS_FORWARD_CXPB_SWEEP_" + current_sweep + "_POINT_POP_CXPB_30_MUTATIONS_16BP_INDMUTATION_10",
            "DBS_FORWARD_CXPB_SWEEP_" + current_sweep + "_POINT_POP_CXPB_30_MUTATIONS_16BP_INDMUTATION_25",
            "DBS_FORWARD_CXPB_SWEEP_" + current_sweep + "_POINT_POP_CXPB_30_MUTATIONS_16BP_INDMUTATION_50",
            "DBS_FORWARD_CXPB_SWEEP_" + current_sweep + "_POINT_POP_CXPB_60_MUTATIONS_16BP_INDMUTATION_10",
            "DBS_FORWARD_CXPB_SWEEP_" + current_sweep + "_POINT_POP_CXPB_60_MUTATIONS_16BP_INDMUTATION_25",
            "DBS_FORWARD_CXPB_SWEEP_" + current_sweep + "_POINT_POP_CXPB_60_MUTATIONS_16BP_INDMUTATION_50",
            "DBS_FORWARD_CXPB_SWEEP_" + current_sweep + "_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_10",
            "DBS_FORWARD_CXPB_SWEEP_" + current_sweep + "_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_25",
            "DBS_FORWARD_CXPB_SWEEP_" + current_sweep + "_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_50"
        ]

    if current_sweep == "5":
        experiment_list = [
            "DBS_FORWARD_CXPB_SWEEP_" + current_sweep + "_POINT_POP_CXPB_30",
            "DBS_FORWARD_CXPB_SWEEP_" + current_sweep + "_POINT_POP_CXPB_60",
            "DBS_FORWARD_CXPB_SWEEP_" + current_sweep + "_POINT_POP_CXPB_90"
        ]

    name_counter += 1
    counter = 0

    print(experiment_list)
    fit_min_floats_average_reps = pd.DataFrame()

    # loop over the 4 metric parameter sweeps (n=3 sweeps)
    for experiments in experiment_list:
        counter += 1

        initial_gen_problem_scores_path = "E:/PhD Files/RQ4/DeBruijnDisruptiveness_Sweeps/" \
                                          + experiment_name + \
                                          experiments + "/metadata/"
        initial_gen_problem_pathway_name = sorted(glob.glob(initial_gen_problem_scores_path + origami_name +
                                                            "*_fits_gen_0.csv"), key=numerical_sort)
        print(initial_gen_problem_pathway_name)

        final_gen_problem_scores_path = "E:/PhD Files/RQ4/DeBruijnDisruptiveness_Sweeps/" \
                                        + experiment_name + \
                                        experiments + "/results/"
        final_gen_problem_pathway_name = sorted(glob.glob(final_gen_problem_scores_path + origami_name +
                                                          "*_fits_gen_10.csv"), key=numerical_sort)
        print(final_gen_problem_pathway_name)

        internal_rep_counter = 0
        # loop over the repetitions logbooks for each sweep (n=10 reps)
        for i in range(len(final_gen_problem_pathway_name)):
            internal_rep_counter += 1

            initial_problem_path_df = pd.read_csv(initial_gen_problem_pathway_name[i])
            final_problem_path_df = pd.read_csv(final_gen_problem_pathway_name[i])
            initial_problem_path_df.drop(columns=["Unnamed: 0"], inplace=True)
            final_problem_path_df.drop(columns=["Unnamed: 0"], inplace=True)

            initial_problem_path_df.columns = ['M1', 'M2', 'M3', 'M4']
            final_problem_path_df.columns = ['M1', 'M2', 'M3', 'M4']
            # get the best metric 1 individual
            initial_individual_df_sorted = initial_problem_path_df.sort_values(by="M1")
            initial_individual_lowest_m1 = initial_individual_df_sorted.iloc[[0]]

            # get the final population of individual and sort by metric 1
            final_individual_df_sorted = final_problem_path_df.sort_values(by="M1")

            # create a percentage improvement table using the two dataframes
            print(initial_individual_lowest_m1)
            final_individual_df_sorted['M1'] = final_individual_df_sorted['M1'].apply(
                lambda x: x - initial_individual_lowest_m1['M1'])
            final_individual_df_sorted['M2'] = final_individual_df_sorted['M2'].apply(
                lambda x: x - initial_individual_lowest_m1['M2'])
            final_individual_df_sorted['M3'] = final_individual_df_sorted['M3'].apply(
                lambda x: x - initial_individual_lowest_m1['M3'])
            final_individual_df_sorted['M4'] = final_individual_df_sorted['M4'].apply(
                lambda x: x - initial_individual_lowest_m1['M4'])

            print(final_individual_df_sorted)
