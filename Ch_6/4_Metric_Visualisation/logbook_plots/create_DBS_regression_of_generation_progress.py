import pandas as pd
import numpy as np
import os
from textwrap import wrap
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
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
            metric_list = ' '.join(strip_i.split()).replace(" ", ",")
            metric_list = metric_list.split(",")
            new_list.append(metric_list)
    return new_list


# all origami list
list_of_origami_names = ["DBS_square",
                         "DBS_square",
                         "DBS_square",
                         "DBS_square",
                         "DBS_square",
                         "DBS_square"]

list_of_origami_names_for_plot = ["DeBruijn Sequence Square Sweep 1",
                                  "DeBruijn Sequence Square Sweep 2",
                                  "DeBruijn Sequence Square Sweep 3",
                                  "DeBruijn Sequence Square Sweep 4",
                                  "DeBruijn Sequence Square Sweep 5",
                                  "DeBruijn Sequence Square Sweep 6"]

# list_of_origami_names = ["DBS_square"]
# list_of_origami_names_for_plot = ["Jurek Square Sweep 4"]

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
    current_sweep = sweep_list[name_counter]

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

        # add a clause for large origami (I.E: nanopore, chineseknot) which will not have a "final" logbook
        if origami_name in large_origami_list:
            problem_scores_path = "E:/PhD Files/RQ4/DeBruijnDisruptiveness_Sweeps/" \
                                  + experiment_name + \
                                  experiments + "/results/"
            problem_pathway_name = sorted(glob.glob(problem_scores_path + origami_name +
                                          "*_logbook_final_gen.csv"), key=numerical_sort)
        # else clause for all other origami sizes
        else:
            problem_scores_path = "E:/PhD Files/RQ4/DeBruijnDisruptiveness_Sweeps/" \
                                  + experiment_name + \
                                  experiments + "/results/"
            problem_pathway_name = sorted(glob.glob(problem_scores_path + origami_name +
                                          "*_logbook_final_gen.csv"), key=numerical_sort)
            print(problem_pathway_name)
        internal_rep_counter = 0
        # loop over the repetitions logbooks for each sweep (n=10 reps)
        for i in problem_pathway_name:
            internal_rep_counter += 1
            problem_path_df = pd.read_csv(i)

            gen = problem_path_df["gen"]
            fit_min = problem_path_df["min"]
            fit_max = problem_path_df["max"]

            # find max float value for whole metric for normalising later
            fit_max_floats = pd.DataFrame(create_list_of_floats(fit_max), columns=["M1", "M2", "M3", "M4"])
            fit_max_floats["M1"] = fit_max_floats.M1.astype(float)
            fit_max_floats["M2"] = fit_max_floats.M2.astype(float)
            fit_max_floats["M3"] = fit_max_floats.M3.astype(float)
            fit_max_floats["M4"] = fit_max_floats.M4.astype(float)

            max_m1 = np.max(fit_max_floats["M1"].values)
            max_m2 = np.max(fit_max_floats["M2"].values)
            max_m3 = np.max(fit_max_floats["M3"].values)
            max_m4 = np.max(fit_max_floats["M4"].values)

            # find min float value for whole metric (important value) across generations
            # CHANGE THIS TO (create_list_of_floats(fit_max) to create MAXIMUM PLOTS:
            fit_min_floats = pd.DataFrame(create_list_of_floats(fit_min), columns=["M1", "M2", "M3", "M4"])
            fit_min_floats["gen"] = problem_path_df["gen"]
            fit_min_floats["rep"] = internal_rep_counter
            fit_min_floats["sweep"] = counter
            fit_min_floats["origami"] = origami_name_for_plot

            fit_min_floats["M1"] = fit_min_floats.M1.astype(float)
            fit_min_floats["M2"] = fit_min_floats.M2.astype(float)
            fit_min_floats["M3"] = fit_min_floats.M3.astype(float)
            fit_min_floats["M4"] = fit_min_floats.M4.astype(float)

            # division of values to normalise them
            # fit_min_floats["M1"] = fit_min_floats["M1"].div(max_m1)
            # fit_min_floats["M2"] = fit_min_floats["M2"].div(max_m2)
            # fit_min_floats["M3"] = fit_min_floats["M3"].div(max_m3)
            # fit_min_floats["M4"] = fit_min_floats["M4"].div(max_m4)

            # concatenate the values to store each repetition metric values per gen across the sweeps.
            fit_min_floats_average_reps = pd.concat([fit_min_floats_average_reps, fit_min_floats])

    print(fit_min_floats_average_reps.head())

    # attempt to create an average here
    # fit_min_floats_average_reps = fit_min_floats_average_reps[[
    #     "M1", "M2", "M3", "M4", "gen", "sweep", "origami"]].groupby(["origami", "sweep", "gen"]).mean()
    f = {"M1": 'mean', "M2": 'mean', "M3": 'mean', "M4": 'mean'}
    fit_min_floats_average_reps = fit_min_floats_average_reps.groupby([
        'origami', 'sweep', 'gen'], as_index=False).agg(f)

    fit_min_floats_all_reps = pd.concat([fit_min_floats_average_reps, fit_min_floats_all_reps])

print(fit_min_floats_all_reps.head(300))

# subplots (multiple 1 plot)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

# limit the sweeps that are shown
fit_min_floats_all_reps = fit_min_floats_all_reps[~fit_min_floats_all_reps['sweep'].isin([2, 3, 4, 5, 6, 7, 8, 9])]

# limit the number of generations
fit_min_floats_all_reps = fit_min_floats_all_reps[fit_min_floats_all_reps['gen'] <= limit_num_of_generations]

# create lower and upper bounds for all of the metrics
metric_1_lower_bound = min(fit_min_floats_all_reps["M1"])  # find LOWER BOUND
metric_1_lower_bound = metric_1_lower_bound - (metric_1_lower_bound / 100 * 2)
metric_1_lower_bound -= metric_1_lower_bound % +0.1 # create lower ceiling
metric_1_upper_bound = max(fit_min_floats_all_reps["M1"])  # find UPPER BOUND
metric_1_upper_bound = metric_1_upper_bound + (metric_1_upper_bound / 100 * 2)
metric_1_upper_bound -= metric_1_upper_bound % -0.1  # create ceiling

# create lower and upper bounds for all of the metrics
metric_2_lower_bound = min(fit_min_floats_all_reps["M2"])  # find LOWER BOUND
metric_2_lower_bound = metric_2_lower_bound - (metric_2_lower_bound / 100 * 2)
metric_2_lower_bound -= metric_2_lower_bound % +0.1  # create lower ceiling
metric_2_upper_bound = max(fit_min_floats_all_reps["M2"])  # find UPPER BOUND
metric_2_upper_bound = metric_2_upper_bound + (metric_2_upper_bound / 100 * 2)
metric_2_upper_bound -= metric_2_upper_bound % -0.1  # create ceiling

# create lower and upper bounds for all of the metrics
metric_3_lower_bound = min(fit_min_floats_all_reps["M3"])  # find LOWER BOUND
metric_3_lower_bound = metric_3_lower_bound - (metric_3_lower_bound / 100 * 2)
metric_3_lower_bound -= metric_3_lower_bound % +0.05  # create lower ceiling
metric_3_upper_bound = max(fit_min_floats_all_reps["M3"])  # find UPPER BOUND
metric_3_upper_bound = metric_3_upper_bound + (metric_3_upper_bound / 100 * 2)
metric_3_upper_bound -= metric_3_upper_bound % -0.05  # create ceiling

# create lower and upper bounds for all of the metrics
metric_4_lower_bound = min(fit_min_floats_all_reps["M4"])  # find LOWER BOUND
metric_4_lower_bound = metric_4_lower_bound - (metric_4_lower_bound / 100 * 2)
metric_4_lower_bound -= metric_4_lower_bound % +0.05  # create lower ceiling
metric_4_upper_bound = max(fit_min_floats_all_reps["M4"])  # find UPPER BOUND
metric_4_upper_bound = metric_4_upper_bound + (metric_4_upper_bound / 100 * 2)
metric_4_upper_bound -= metric_4_upper_bound % -0.05  # create ceiling

for key, grp in fit_min_floats_all_reps.groupby(['origami']):
    ax1 = grp.plot(ax=ax1, kind='line', x='gen', y='M1', label=key,
                   ylim=(metric_1_lower_bound, metric_1_upper_bound),
                   xlim=(0, limit_num_of_generations),
                   ylabel="Metric 1", xlabel="Generation", linewidth=3.0)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="upper right")

for key, grp in fit_min_floats_all_reps.groupby(['origami']):
    ax2 = grp.plot(ax=ax2, kind='line', x='gen', y='M2', label=key,
                   ylim=(metric_2_lower_bound, metric_2_upper_bound),
                   xlim=(0, limit_num_of_generations),
                   ylabel="Metric 2", xlabel="Generation", linewidth=3.0)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc="upper right")

for key, grp in fit_min_floats_all_reps.groupby(['origami']):
    ax3 = grp.plot(ax=ax3, kind='line', x='gen', y='M3', label=key,
                   ylim=(metric_3_lower_bound, metric_3_upper_bound),
                   xlim=(0, limit_num_of_generations),
                   ylabel="Metric 3", xlabel="Generation", linewidth=3.0)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels, loc="upper right")

for key, grp in fit_min_floats_all_reps.groupby(['origami']):
    ax4 = grp.plot(ax=ax4, kind='line', x='gen', y='M4', label=key,
                   ylim=(metric_4_lower_bound, metric_4_upper_bound),
                   xlim=(0, limit_num_of_generations),
                   ylabel="Metric 4", xlabel="Generation", linewidth=3.0)
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles, labels, loc="upper right")

experiment_name_for_plot = plot_name + " Minimum Fitness Metric Average Repetitions " \
                                       "Genetic Algorithm Parameter Sweep Set 1"

title = ("\n".join(wrap(experiment_name_for_plot, 50)))
fig.suptitle(title, size=20)
fig.set_dpi(300)
plt.tight_layout()
# plt.draw()
fig.savefig(path_parent + "/Logbook_Plots/logbook_minimum_fitness_plots/" + save_name +
            "DBS_Initial_Minimum_Fitness_Metric_GA_All_Repetitions_Sweep_Set_1_All_Repetitions" + ".png",
            bbox_inches='tight', format='png')
# plt.show()
