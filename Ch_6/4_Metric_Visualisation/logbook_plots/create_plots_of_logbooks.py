import pandas as pd
import numpy as np
from os import path
from textwrap import wrap
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
from topsis import topsis
import topsispy as tp  # implementation that is used, doesn't matter which is used
import glob
import re
import matplotlib.pyplot as plt


plt.style.use('ggplot')


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def create_list_of_floats(series):
    new_list = []
    for i in series:
        metric_list = i.replace(" ", ",").strip("]").strip("[").split(",")
        new_list.append(metric_list)
    return new_list


# create list of experiment names
experiment_list = [
                   "CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_10",
                   "CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_25",
                   "CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_50",
                   ]

origami_name = "capsule-linear"
experiment_set_name = origami_name + "_SWEEP/"


counter = 0
for experiments in experiment_list:
    counter += 1
    problem_scores_path = "D:/Generator_RQ4/Systematic_Experiments/" + origami_name + "_SWEEP/" + \
                          experiments + "/results/"
    problem_pathway_name = sorted(glob.glob(problem_scores_path + origami_name +
                                  "*_logbook_final_gen.csv"), key=numerical_sort)

    # store all reps for an experiment
    fit_min_floats_all_reps = pd.DataFrame()

    internal_rep_counter = 0
    for i in problem_pathway_name:
        internal_rep_counter += 1
        problem_path_df = pd.read_csv(i)

        gen = problem_path_df["gen"]
        fit_min = problem_path_df["min"]

        fit_min_floats = pd.DataFrame(create_list_of_floats(fit_min), columns=["M1", "M2", "M3", "M4"])
        fit_min_floats["gen"] = problem_path_df["gen"]
        fit_min_floats["rep"] = internal_rep_counter

        fit_min_floats["M1"] = fit_min_floats.M1.astype(float)
        fit_min_floats["M2"] = fit_min_floats.M2.astype(float)
        fit_min_floats["M3"] = fit_min_floats.M3.astype(float)
        fit_min_floats["M4"] = fit_min_floats.M4.astype(float)

        fit_min_floats_all_reps = pd.concat([fit_min_floats_all_reps, fit_min_floats])

        # fig, ax = plt.subplots()
        # ax.scatter(fit_min_floats["gen"], fit_min_floats["M1"])
        # ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        # plt.show()

    fig, axes = plt.subplots(4)

    fit_min_floats_all_reps["M1"] = fit_min_floats_all_reps.M1.astype(float)
    fit_min_floats_all_reps["M2"] = fit_min_floats_all_reps.M2.astype(float)
    fit_min_floats_all_reps["M3"] = fit_min_floats_all_reps.M3.astype(float)
    fit_min_floats_all_reps["M4"] = fit_min_floats_all_reps.M4.astype(float)

    fit_min_floats_all_reps = fit_min_floats_all_reps.sort_values("gen")

    fig.suptitle(origami_name + " Minimum Fitness Metrics GA All Repetitions Sweep Set " + str(counter), size=10)
    metric_1_lower_bound = min(fit_min_floats_all_reps["M1"])  # find LOWER BOUND
    metric_1_lower_bound = metric_1_lower_bound - (metric_1_lower_bound / 100 * 2)
    metric_1_lower_bound -= metric_1_lower_bound % +1000  # create lower ceiling
    metric_1_upper_bound = max(fit_min_floats_all_reps["M1"])  # find UPPER BOUND
    metric_1_upper_bound = metric_1_upper_bound + (metric_1_upper_bound / 100 * 2)
    metric_1_upper_bound -= metric_1_upper_bound % -1000  # create ceiling

    axes[0].plot(fit_min_floats_all_reps["gen"], fit_min_floats_all_reps["M1"])
    axes[0].yaxis.set_major_locator(plt.MaxNLocator(5))
    axes[0].set_title('Metric 1', size='8')
    axes[0].set_ylim(metric_1_lower_bound, metric_1_upper_bound)

    metric_2_lower_bound = min(fit_min_floats_all_reps["M2"])  # find LOWER BOUND
    metric_2_lower_bound = metric_2_lower_bound - (metric_2_lower_bound / 100 * 5)
    metric_2_lower_bound -= metric_2_lower_bound % +1000  # create lower ceiling
    metric_2_upper_bound = max(fit_min_floats_all_reps["M2"])  # find UPPER BOUND
    metric_2_upper_bound = metric_2_upper_bound + (metric_2_upper_bound / 100 * 5)
    metric_2_upper_bound -= metric_2_upper_bound % -1000  # create ceiling
    axes[1].plot(fit_min_floats_all_reps["gen"], fit_min_floats_all_reps["M2"])
    axes[1].yaxis.set_major_locator(plt.MaxNLocator(5))
    axes[1].set_title('Metric 2', size='8')
    axes[1].set_ylim(metric_2_lower_bound, metric_2_upper_bound)

    metric_3_lower_bound = min(fit_min_floats_all_reps["M3"])  # find LOWER BOUND
    metric_3_lower_bound = metric_3_lower_bound - (metric_3_lower_bound / 100 * 2)
    metric_3_lower_bound -= metric_3_lower_bound % +1  # create lower ceiling
    metric_3_upper_bound = max(fit_min_floats_all_reps["M3"])  # find UPPER BOUND
    metric_3_upper_bound = metric_3_upper_bound + (metric_3_upper_bound / 100 * 2)
    metric_3_upper_bound -= metric_3_upper_bound % -1  # create ceiling
    axes[2].plot(fit_min_floats_all_reps["gen"], fit_min_floats_all_reps["M3"])
    axes[2].yaxis.set_major_locator(plt.MaxNLocator(5))
    axes[2].set_title('Metric 3', size='8')
    axes[2].set_ylim(metric_3_lower_bound, metric_3_upper_bound)

    metric_4_lower_bound = min(fit_min_floats_all_reps["M4"])  # find LOWER BOUND
    metric_4_lower_bound = metric_4_lower_bound - (metric_4_lower_bound / 100 * 2)
    metric_4_lower_bound -= metric_4_lower_bound % +100  # create lower ceiling
    metric_4_upper_bound = max(fit_min_floats_all_reps["M4"])  # find UPPER BOUND
    metric_4_upper_bound = metric_4_upper_bound + (metric_4_upper_bound / 100 * 2)
    metric_4_upper_bound -= metric_4_upper_bound % -100  # create ceiling
    axes[3].plot(fit_min_floats_all_reps["gen"], fit_min_floats_all_reps["M4"])
    axes[3].yaxis.set_major_locator(plt.MaxNLocator(5))
    axes[3].set_title('Metric 4', size='8')
    axes[3].set_ylim(metric_4_lower_bound, metric_4_upper_bound)
    fig.set_dpi(100)
    plt.show()

    import time
    time.sleep(5)
