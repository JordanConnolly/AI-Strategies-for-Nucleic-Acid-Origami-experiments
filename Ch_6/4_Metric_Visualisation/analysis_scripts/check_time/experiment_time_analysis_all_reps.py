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

origami_name = "8__dunn"
experiment_set_name = origami_name + "_SWEEP/"
# create variable for num of individual in gen
individuals = 40
median_value = 4
# calculate the best alternative using TOPSIS implementation
weights = [0.25, 0.25, 0.25, 0.25]
costs = [-1, -1, -1, -1]

# create performance indicator for use later
hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2, 1.2, 1.2]))

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


sum_time_list = []
time_list = []
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

        gen_0_time_file_name = "_GA-experiment_rep_" + str(rep) + "_gen_time.txt"
        all_rep_gen_0_time_df = pd.read_csv(metadata_path + origami_name + gen_0_time_file_name, header=None)

        print(all_rep_gen_0_time_df)
        time_file_name = "_GA-experiment_rep_" + str(rep) + \
                         "_rep_" + str(rep) + "_gen_time.txt"
        all_rep_gen_time_df = pd.read_csv(metadata_path + origami_name + time_file_name, header=None)
        print(all_rep_gen_time_df)

        for time in all_rep_gen_time_df.itertuples():
            gen_time = time[1].split(":")
            time_list.append(float(gen_time[1]))


average_gen_time = np.average(time_list)
sum_gen_time = np.sum(time_list)
print("average gen times:  ", average_gen_time)
print("sum gen times:  ", sum_gen_time / 30)
