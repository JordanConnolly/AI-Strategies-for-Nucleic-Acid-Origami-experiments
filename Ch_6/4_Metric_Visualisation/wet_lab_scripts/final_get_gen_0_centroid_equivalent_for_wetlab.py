from pymoo.factory import get_problem, get_reference_directions
from pymoo.problems.many.dtlz import DTLZ1
from pymoo.visualization.pcp import PCP
from pymoo.factory import get_performance_indicator
import pandas as pd
import numpy as np
from os import path
from textwrap import wrap
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
from os import path
import glob
from pymoo.factory import get_performance_indicator
from textwrap import wrap
import topsispy as tp
import re


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# base_path / path to all files in this project
basepath = path.dirname(__file__)

# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # SET EXPERIMENT PARAMETERS # # # # #
meta_data_list = []
origami_name = "DBS_square"
experiment_set_name = origami_name + "_SWEEP/"

# create list of experiment names
experiment_list = [
    "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_10",
    "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_25",
    "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_50",
]


def is_pareto(costs, maximise=None):
    # source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    :param costs: An (n_points, n_costs) array
    :maximise: boolean. True for maximising, False for minimising
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)  # Remove dominated points
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)  # Remove dominated points
    return is_efficient


# store the pareto fronts here
merged_fwd_scaffold_and_scores_df = pd.DataFrame()

for forward_experiments in experiment_list:
    # access externally
    filepath = "E:/PhD Files/RQ4/4_Metrics_SWEEP_Final_Energy_Model/" + experiment_set_name + forward_experiments
    input_path = filepath  # find the path for the input data
    metadata_path = input_path + "/metadata/"  # specify the metadata path
    results_path = input_path + "/results/"  # specify the results path
    plot_path = input_path + "/plots/"  # specify the plots path

    for rep in range(1, 11, 1):
        print(metadata_path + str(rep))
        # create a data frame of the scaffolds from median repetition
        forward_gen0_scaffold_df = pd.read_csv(metadata_path + origami_name + "_GA-experiment_rep_" +
                                               str(rep) + "_scaffolds_gen_0.csv")
        forward_gen0_scaffold_df.drop(columns=["Unnamed: 0"], inplace=True)

        # all gens in loop
        rep_gen0_score_df = pd.read_csv(metadata_path + origami_name + "_GA-experiment_rep_" +
                                        str(rep) + "_fits_gen_0.csv")
        rep_gen0_score_df.drop(columns=["Unnamed: 0"], inplace=True)

        inner_merged_fwd_scaffold_and_scores_df = pd.merge(rep_gen0_score_df,
                                                           forward_gen0_scaffold_df, left_index=True, right_index=True)
        merged_fwd_scaffold_and_scores_df = pd.concat([merged_fwd_scaffold_and_scores_df,
                                                       inner_merged_fwd_scaffold_and_scores_df])

print(merged_fwd_scaffold_and_scores_df.shape)

merged_fwd_scaffold_and_scores_df.columns = ["Metric 1",
                                             "Metric 2", "Metric 3",
                                             "Metric 4", "scaffold_sequence"]
print(merged_fwd_scaffold_and_scores_df)
print(merged_fwd_scaffold_and_scores_df.shape)

# duped = merged_fwd_scaffold_and_scores_df[merged_fwd_scaffold_and_scores_df.duplicated(["scaffold_sequence"])==True]
# grouped_by_duped = duped.groupby('scaffold_sequence').count().sort_values(by=["Metric 1"], ascending=False)
# print(grouped_by_duped.head(15))

# remove duplicate metrics rows
merged_fwd_scaffold_and_scores_df.drop_duplicates(keep='first', subset=["scaffold_sequence"],
                                                  inplace=True)
print(merged_fwd_scaffold_and_scores_df.shape)

# create entire pool index
merged_fwd_scaffold_and_scores_df = merged_fwd_scaffold_and_scores_df.rename_axis('in_rep_index').reset_index()
merged_fwd_scaffold_and_scores_df.drop(columns=["in_rep_index"],
                                       inplace=True)
merged_fwd_scaffold_and_scores_df = merged_fwd_scaffold_and_scores_df.rename_axis('entire_pool_index').reset_index()

print(len(merged_fwd_scaffold_and_scores_df))

merged_fwd_scaffold_and_scores_df.to_csv("PARETO_SELECTED_SEQUENCES/" + origami_name +
                                         "_gen0_forward_set_pareto_not_selected.csv")

merged_fwd_scaffold_and_scores_df["scaffold_sequence"].to_csv("PARETO_SELECTED_SEQUENCES/" + origami_name +
                                                              "_SCAFFOLD_SEQUENCES_ONLY_gen0_forward_set_"
                                                              "_gen0_forward_set_unique_scaffolds_not_pareto_front.csv")

# apply function to gather pareto front scores
fwd_pool_dropped = merged_fwd_scaffold_and_scores_df[["Metric 1", "Metric 2", "Metric 3", "Metric 4"]]
fwd_pool_scores = fwd_pool_dropped.to_numpy(dtype=np.float64)
fwd_pool_pareto = is_pareto(fwd_pool_scores, maximise=False)
fwd_pool_pareto_front = fwd_pool_scores[fwd_pool_pareto]
fwd_pool_pareto_front_df = pd.DataFrame(fwd_pool_pareto_front)
fwd_pool_pareto_front = fwd_pool_pareto_front_df.values

fwd_pool_index = merged_fwd_scaffold_and_scores_df
fwd_pool_pareto_with_index_and_scaffold_score = pd.DataFrame()
for individual in fwd_pool_pareto_front:
    mask_1 = fwd_pool_index['Metric 1'].isin([individual[0]])
    mask_2 = fwd_pool_index['Metric 2'].isin([individual[1]])
    mask_3 = fwd_pool_index['Metric 3'].isin([individual[2]])
    mask_4 = fwd_pool_index['Metric 4'].isin([individual[3]])
    mask_concat = pd.concat((mask_1, mask_2, mask_3, mask_4), axis=1)
    mask = mask_concat.all(axis=1)
    fwd_pool_selected_individual_with_index = fwd_pool_index[mask]
    fwd_pool_pareto_with_index_and_scaffold_score = pd.concat([fwd_pool_pareto_with_index_and_scaffold_score,
                                                               fwd_pool_selected_individual_with_index])

print(fwd_pool_pareto_with_index_and_scaffold_score.columns)

fwd_pool_pareto_with_index_and_scaffold_score.columns = ["entire_pool_index",
                                                         "Metric 1",
                                                         "Metric 2", "Metric 3",
                                                         "Metric 4", "scaffold_sequence"]
print(fwd_pool_pareto_with_index_and_scaffold_score.shape)

# # remove duplicate metrics rows
# merged_fwd_scaffold_and_scores_df.drop_duplicates(keep='first', subset=["scaffold_sequence"],
#                                                   inplace=True)
# print(len(fwd_pool_pareto_with_index_and_scaffold_score))

# reset the index (unnamed: 0 column)
fwd_pool_pareto_with_index_and_scaffold_score.reset_index(drop=True, inplace=True)
print(fwd_pool_pareto_with_index_and_scaffold_score.head(5))


fwd_pool_pareto_with_index_and_scaffold_score.to_csv("PARETO_SELECTED_SEQUENCES/" + origami_name +
                                                     "_gen0_forward_set_pareto_selected.csv")

fwd_pool_pareto_with_index_and_scaffold_score["scaffold_sequence"].to_csv("PARETO_SELECTED_SEQUENCES/" + origami_name +
                                                                          "_SCAFFOLD_SEQUENCES_ONLY_gen0_forward_sets_"
                                                                          "pareto_front_of_final_sequences_selected_"
                                                                          "code.csv")
