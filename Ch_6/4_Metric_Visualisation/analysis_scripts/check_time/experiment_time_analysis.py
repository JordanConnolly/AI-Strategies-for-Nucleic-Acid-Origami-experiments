from pymoo.factory import get_problem, get_reference_directions
from pymoo.problems.many.dtlz import DTLZ1
from pymoo.visualization.pcp import PCP
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_performance_indicator
import pandas as pd
import numpy as np
from os import path
from textwrap import wrap
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
from os import path
import re
import glob
from pymoo.factory import get_performance_indicator
from textwrap import wrap


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

# create list of experiment names
experiment_list = [
                   "CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_10",
                   "CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_25",
                   "CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_50"
                   ]

origami_name = "10__A-brick-1512"
experiment_set_name = origami_name + "_SWEEP/"
# create variable for num of individual in gen
individuals = 40
median_value = 4
# calculate the best alternative using TOPSIS implementation
weights = [0.25, 0.25, 0.25, 0.25]
costs = [-1, -1, -1, -1]

# create performance indicator for use later
hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2, 1.2, 1.2]))

time_list = []


"""
SOURCE: https://pymoo.org/visualization/pcp.html
"""

# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # SET EXPERIMENT PARAMETERS # # # # #


def is_pareto(costs, maximise=False):
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


for experiment_name in experiment_list:
    # access externally
    filepath = "D:/Generator_RQ4/Systematic_Experiments/" + experiment_set_name + experiment_name
    input_path = filepath  # find the path for the input data
    print("\n", input_path)
    metadata_path = input_path + "/metadata/"  # specify the metadata path
    results_path = input_path + "/results/"  # specify the results path
    plot_path = input_path + "/plots/"  # specify the plots path
    hyper_volumes = []

    for i in range(1, 11, 1):
        rep = i
        if rep is not 35:
            # all gens in loop
            all_gen_score_df = pd.read_csv(results_path + origami_name + "_GA-experiment_rep_" +
                                           str(rep) + "_scores_final_gen.csv")  # load all of the generation score
            all_gen_scaffold_df = pd.read_csv(results_path + origami_name + "_GA-experiment_rep_" +
                                              str(rep) + "_scaffolds_final_gen.csv")
            # print all gens
            # print(all_gen_scaffold_df.iloc[0])
            # print(all_gen_score_df.iloc[0])
            # calculate number of generations from score
            generations = len(all_gen_score_df) / individuals

            # create generation column list for scores
            score_gen_list = []
            ind_counter = 0
            gen_counter = 0
            for k in range(len(all_gen_score_df)):
                ind_counter += 1
                score_gen_list.append(gen_counter)
                if ind_counter == individuals:
                    ind_counter = 0
                    gen_counter += 1

            final_gen = max(score_gen_list)
            # assign the generation column for filtering
            all_gen_score_df['generation'] = score_gen_list
            # print(all_gen_score_df)

            # create normalised scores of the pareto front / final gen
            gen_scores = all_gen_score_df.loc[all_gen_score_df['generation'] == final_gen].drop(columns=['Unnamed: 0',
                                                                                                         'generation'
                                                                                                         ])
            scores = gen_scores.to_numpy(dtype=np.float64)
            # apply function to gather pareto front scores
            pareto = is_pareto(scores)
            pareto_front = scores[pareto]
            pareto_front_df = pd.DataFrame(pareto_front)
            pareto_front_df.sort_values(0, inplace=True)
            pareto_front = pareto_front_df.values
            hv_calculated = hv.calc(scores)
            hyper_volumes.append([experiment_name, rep, hv_calculated])

    # create a df to analyse and get best/median, before adding best and median to total experiment df
    hyper_volume_df = pd.DataFrame(hyper_volumes)
    hyper_volume_df.columns = ["experiment", "rep", "hypervolume"]
    hyper_volume_df.sort_values(by=['hypervolume'], inplace=True, ascending=False)

    med_rep_df = hyper_volume_df.iloc[median_value, :].copy()

    gen_0_time_file_name = "_GA-experiment_rep_" + str(med_rep_df["rep"]) + "_gen_time.txt"
    med_rep_gen_0_time_df = pd.read_csv(metadata_path + origami_name + gen_0_time_file_name, header=None)
    # med_rep_gen_0_time_df.columns = ["Time"]

    print(med_rep_gen_0_time_df)
    time_file_name = "_GA-experiment_rep_" + str(med_rep_df["rep"]) + \
                     "_rep_" + str(med_rep_df["rep"]) + "_gen_time.txt"
    med_rep_gen_time_df = pd.read_csv(metadata_path + origami_name + time_file_name, header=None)
    # med_rep_gen_time_df.columns = ["Time"]
    print(med_rep_gen_time_df)

    for time in med_rep_gen_time_df.itertuples():
        gen_time = time[1].split(":")
        time_list.append(float(gen_time[1]))

print(time_list)
average_gen_time = np.average(time_list)
sum_gen_time = np.sum(time_list)
print("average gen times:  ", average_gen_time)
print("sum gen times:  ", sum_gen_time / 3)
