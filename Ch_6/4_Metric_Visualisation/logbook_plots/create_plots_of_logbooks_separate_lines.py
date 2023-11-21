import pandas as pd
import numpy as np
import os
from textwrap import wrap
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
from topsis import topsis
import topsispy as tp  # implementation that is used, doesn't matter which is used
import glob
import re
import matplotlib.pyplot as plt

# print(plt.style.available)
# plt.style.use('ggplot')  # best so far
plt.style.use('seaborn-whitegrid')
# plt.style.use('fivethirtyeight')
# plt.style.use('bmh')

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
            print(strip_i)
            metric_list = ' '.join(strip_i.split()).replace(" ", ",")
            metric_list = metric_list.split(",")
            new_list.append(metric_list)
            print(metric_list)
    return new_list


# create list of experiment names
experiment_list = [
                   "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_10",
                   "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_25",
                   "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_50"
                   ]

# for minitri use the RERUN.CSV for logbook  # only on older 4 metric work

list_of_origami_names = ['ball', '6hb', 'dunn', 'DBS_square', 'minitri', 'Abrick', 'fourfinger-circular',
                         'fourfinger-linear', 'nanoribbonRNA', 'hj']

list_of_origami_names_for_plot = ["Single HJ",
                                  "Nanoribbon (RNA)",
                                  "M1.3 Four Finger (Linear)",
                                  "M1.3 Four Finger (Circular)",
                                  "Mini Triangle",
                                  "DeBruijn Sequence Square",
                                  "Solid Brick",
                                  "6 Helix Bundle",
                                  "Ball",
                                  "Dannenberg Tile with TTTT overhangs"]

large_origami_list = []
plot_name = "4 Metric Origami Results Scaled"
save_name = "4_Metric_Origami_Results_Scaled_"
limit_num_of_generations = 251

# loop over logbook, get the min fit for all gens, plot as a line
# now we need to loop over origamis, store their logbook as separate groups, plot lines

# store all reps for an experiment
fit_min_floats_all_reps = pd.DataFrame()

# for origami_name in list_of_origami_names:
name_counter = 0
for origami_name in list_of_origami_names:
    experiment_set_name = origami_name + "_SWEEP/"
    print(experiment_set_name)
    origami_name_for_plot = list_of_origami_names_for_plot[name_counter]
    name_counter += 1
    counter = 0
    fit_min_floats_average_reps = pd.DataFrame()

    # loop over the 4 metric parameter sweeps (n=3 sweeps)
    for experiments in experiment_list:
        counter += 1

        # add a clause for large origami (I.E: nanopore, chineseknot) which will not have a "final" logbook
        if origami_name in large_origami_list:
            problem_scores_path = "E:/PhD Files/RQ4/4_Metrics_SWEEP_Final_Energy_Model/" + origami_name + "_SWEEP/" + \
                                  experiments + "/metadata/"
            problem_pathway_name = sorted(glob.glob(problem_scores_path + origami_name +
                                          "*_logbook_gen.csv"), key=numerical_sort)

        # else clause for all other origami sizes
        else:
            problem_scores_path = "E:/PhD Files/RQ4/4_Metrics_SWEEP_Final_Energy_Model/" + origami_name + "_SWEEP/" + \
                                  experiments + "/results/"
            problem_pathway_name = sorted(glob.glob(problem_scores_path + origami_name +
                                          "*_logbook_final_gen.csv"), key=numerical_sort)

        internal_rep_counter = 0
        for i in problem_pathway_name:
            print(internal_rep_counter)
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

        # subplots (multiple 1 plot)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

        for key, grp in fit_min_floats_all_reps.groupby(['rep']):
            metric_1_lower_bound = min(fit_min_floats_all_reps["M1"])  # find LOWER BOUND
            metric_1_lower_bound = metric_1_lower_bound - (metric_1_lower_bound / 100 * 2)
            metric_1_lower_bound -= metric_1_lower_bound % +1000  # create lower ceiling
            metric_1_upper_bound = max(fit_min_floats_all_reps["M1"])  # find UPPER BOUND
            metric_1_upper_bound = metric_1_upper_bound + (metric_1_upper_bound / 100 * 2)
            metric_1_upper_bound -= metric_1_upper_bound % -1000  # create ceiling
            ax1 = grp.plot(ax=ax1, kind='line', x='gen', y='M1', label=key,
                           ylim=(metric_1_lower_bound, metric_1_upper_bound),
                           ylabel="Metric 1", xlabel="Generation")

        for key, grp in fit_min_floats_all_reps.groupby(['rep']):
            metric_2_lower_bound = min(fit_min_floats_all_reps["M2"])  # find LOWER BOUND
            metric_2_lower_bound = metric_2_lower_bound - (metric_2_lower_bound / 100 * 5)
            metric_2_lower_bound -= metric_2_lower_bound % +1000  # create lower ceiling
            metric_2_upper_bound = max(fit_min_floats_all_reps["M2"])  # find UPPER BOUND
            metric_2_upper_bound = metric_2_upper_bound + (metric_2_upper_bound / 100 * 5)
            metric_2_upper_bound -= metric_2_upper_bound % -1000  # create ceiling
            ax2 = grp.plot(ax=ax2, kind='line', x='gen', y='M2', label=key,
                           ylim=(metric_2_lower_bound, metric_2_upper_bound),
                           ylabel="Metric 2", xlabel="Generation")

        for key, grp in fit_min_floats_all_reps.groupby(['rep']):
            metric_3_lower_bound = min(fit_min_floats_all_reps["M3"])  # find LOWER BOUND
            metric_3_lower_bound = metric_3_lower_bound - (metric_3_lower_bound / 100 * 2)
            metric_3_lower_bound -= metric_3_lower_bound % +5  # create lower ceiling
            metric_3_upper_bound = max(fit_min_floats_all_reps["M3"])  # find UPPER BOUND
            metric_3_upper_bound = metric_3_upper_bound + (metric_3_upper_bound / 100 * 2)
            metric_3_upper_bound -= metric_3_upper_bound % -5  # create ceiling
            ax3 = grp.plot(ax=ax3, kind='line', x='gen', y='M3', label=key,
                           ylim=(metric_3_lower_bound, metric_3_upper_bound),
                           ylabel="Metric 3", xlabel="Generation")

        for key, grp in fit_min_floats_all_reps.groupby(['rep']):
            metric_4_lower_bound = min(fit_min_floats_all_reps["M4"])  # find LOWER BOUND
            metric_4_lower_bound = metric_4_lower_bound - (metric_4_lower_bound / 100 * 2)
            metric_4_lower_bound -= metric_4_lower_bound % +5  # create lower ceiling
            metric_4_upper_bound = max(fit_min_floats_all_reps["M4"])  # find UPPER BOUND
            metric_4_upper_bound = metric_4_upper_bound + (metric_4_upper_bound / 100 * 2)
            metric_4_upper_bound -= metric_4_upper_bound % -5  # create ceiling
            ax4 = grp.plot(ax=ax4, kind='line', x='gen', y='M4', label=key,
                           ylim=(metric_4_lower_bound, metric_4_upper_bound),
                           ylabel="Metric 4", xlabel="Generation")

        fig.suptitle(plot_name + " Minimum Fitness Metric All Repetitions Genetic Algorithm Parameter Sweep Set "
                     + str(counter), size=20)
        fig.set_dpi(300)
        plt.tight_layout()
        plt.draw()
        fig.savefig(path_parent + "/Logbook_Plots/logbook_minimum_fitness_plots/" + save_name +
                    "Minimum_Fitness_Metric_GA_All_Repetitions_Sweep_Set_" + str(counter) + ".png",
                    bbox_inches='tight', format='png')
        plt.show()
