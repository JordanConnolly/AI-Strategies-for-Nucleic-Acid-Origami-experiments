from pymoo.factory import get_performance_indicator
import pandas as pd
import numpy as np
from os import path
from textwrap import wrap
# -*- coding: utf-8 -*-
from GRA_metric import GrayRelationalCoefficient, plot_average_grey_relational_coefficient
from topsis import topsis
import topsispy as tp  # implementation that is used, doesn't matter which is used
import glob

# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # SET EXPERIMENT PARAMETERS # # # # #
import re


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# create list of experiment names
experiment_list = [
                   "CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_10",
                   "CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_25",
                   "CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_50",
                   ]

list_of_origami_names = ['ball', '6hb', 'dunn', 'DBS_square',
                         'minitri', 'Abrick', 'fourfinger-circular',
                         'fourfinger-linear',
                         'nanoribbonRNA', 'hj']

origami_name = "ball"
experiment_set_name = origami_name + "_SWEEP/"
# create variable for num of individual in gen
individuals = 40
median_value = 4

# access and import the random search scores
random_scores_path = "E:/PhD Files/RQ4/4_Metrics_SWEEP_Final_Energy_Model/" + origami_name + \
                     "_SWEEP/Random_Walk_300000/metadata/"
random_score_pathway_names = sorted(glob.glob(random_scores_path + "*.csv"), key=numerical_sort)

ten_random_score_pathway_names = []
counter = 0
for pathways in random_score_pathway_names:
    counter += 1
    if counter < 11:
        ten_random_score_pathway_names.append(pathways)
random_pool_df = pd.concat((pd.read_csv(f) for f in ten_random_score_pathway_names))
print(random_pool_df)

drop_list = ["0.0", "1.0", "2.0", "3.0"]
random_pool_df = random_pool_df[~random_pool_df.isin(drop_list)]
random_pool_df.fillna(0, inplace=True)
new_worst_random_pool = []
new_worst_random_pool.append(max(random_pool_df.iloc[:, 1].values))
new_worst_random_pool.append(max(random_pool_df.iloc[:, 2].values))
new_worst_random_pool.append(max(random_pool_df.iloc[:, 3].values))
new_worst_random_pool.append(max(random_pool_df.iloc[:, 4].values))
print(new_worst_random_pool)
worst_random_pool_df = pd.DataFrame(new_worst_random_pool)

# normalisation values for metrics
worst_theoretical_individual = new_worst_random_pool  # 100,000 seq random pool
print(worst_theoretical_individual)

drop_list = ["0.0", "1.0", "2.0", "3.0"]
random_pool_df = random_pool_df[~random_pool_df.isin(drop_list)]
random_pool_df.fillna(0, inplace=True)
new_best_random_pool = []
new_best_random_pool.append(min(random_pool_df.iloc[:, 1].values))
new_best_random_pool.append(min(random_pool_df.iloc[:, 2].values))
new_best_random_pool.append(min(random_pool_df.iloc[:, 3].values))
new_best_random_pool.append(min(random_pool_df.iloc[:, 4].values))
print(new_best_random_pool)
best_random_pool_df = pd.DataFrame(new_worst_random_pool)

# normalisation values for metrics
best_theoretical_individual = new_best_random_pool  # 100,000 seq random pool
print(best_theoretical_individual)

# create performance indicator for use later
hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2, 1.2, 1.2]))

# calculate the best alternative using TOPSIS implementation
weights = [0.25, 0.25, 0.25, 0.25]
costs = [-1, -1, -1, -1]

# access and import the reverse problem origami set scores
reverse_problem_dfs = pd.DataFrame()
concatenated_reverse_problem_max = pd.DataFrame()

for reverse_experiments in experiment_list:
    reverse_problem_scores_path = "D:/Generator_RQ4/Systematic_Experiments/" + origami_name + "_SWEEP/BACKWARDS_" + \
                                  reverse_experiments + "/results/"
    reverse_problem_pathway_name = sorted(glob.glob(reverse_problem_scores_path + origami_name +
                                                    "*_logbook_final_gen_rerun.csv"), key=numerical_sort)
    reverse_problem_df = pd.concat((pd.read_csv(f) for f in reverse_problem_pathway_name))
    reverse_problem_dfs = pd.concat([reverse_problem_dfs, reverse_problem_df])
    reverse_problem_max = reverse_problem_dfs.loc[reverse_problem_dfs['gen'] == 250]
    concatenated_reverse_problem_max = pd.concat([concatenated_reverse_problem_max, reverse_problem_max])

max_list = []
for max_reverse_values in concatenated_reverse_problem_max["max"]:
    max_values_list = max_reverse_values.replace(" ", ",")
    max_values_list = max_values_list.strip("[").strip("]")
    max_values_split = max_values_list.split(",")
    new_list = []
    for value in max_values_split:
        new_list.append(float(value))
    max_list.append(new_list)

reverse_problem_wti_df = pd.DataFrame(max_list)
reverse_theoretical_individual = [max(reverse_problem_wti_df.iloc[:, 0].values),
                                  max(reverse_problem_wti_df.iloc[:, 1].values),
                                  max(reverse_problem_wti_df.iloc[:, 2].values),
                                  max(reverse_problem_wti_df.iloc[:, 3].values)]
print("new theoretical reverse individual:", reverse_theoretical_individual)
# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # #

# create a data frame to store hyper-volume analysis results for all experiments in list
all_hyper_volumes_df = pd.DataFrame()
# create a data frame to store GRA (extra metric)
all_gra_df = pd.DataFrame()


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


# def is_pareto(scores):
#     # Count number of items
#     population_size = scores.shape[0]
#     # Create a NumPy index for scores on the pareto front (zero indexed)
#     population_ids = np.arange(population_size)
#     # Create a starting list of items on the Pareto front
#     # All items start off as being labelled as on the Pareto front
#     pareto_front = np.ones(population_size, dtype=bool)
#     # Loop through each item. This will then be compared with all other items
#     for i in range(population_size):
#         # Loop through all other items
#         for j in range(population_size):
#             # Check if our 'i' point is dominated by out 'j' point
#             if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
#                 # j dominates i. Label 'i' point as not on Pareto front
#                 pareto_front[i] = 0
#                 # Stop further comparisons with 'i' (no more comparisons needed)
#                 break
#     # Return ids of scenarios on pareto front
#     return population_ids[pareto_front]
# # Fairly fast for many datapoints, less fast for many costs, somewhat readable


# def is_pareto(costs):
#     """
#     Find the pareto-efficient points
#     :param costs: An (n_points, n_costs) array
#     :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
#     """
#     is_efficient = np.ones(costs.shape[0], dtype=bool)
#     for i, c in enumerate(costs):
#         if is_efficient[i]:
#             is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)  # Keep any point with a lower cost
#             is_efficient[i] = True  # And keep self
#     return is_efficient


def un_normalise_metrics(individual, worst_random_pool_metrics):
    # for presentation purposes: un-normalised metrics
    individual_metrics = []
    counter = 0
    for i in range(len(individual)):
        if counter == i:
            un_norm_metric = worst_random_pool_metrics[i] * individual[i]
            individual_metrics.append(un_norm_metric)
        counter += 1
    return individual_metrics


# normalise results to between 0 and 1 using worst random pool values,
# use worst theoretical individual normalised data-frame and find the pareto front for ind selection
random_pool_df_wti_norm = random_pool_df.copy()
# normalise results to between 0 and 1 using worst random pool values possible

random_pool_df_wti_norm["0"] = reverse_theoretical_individual[0] - random_pool_df["0"].values + best_theoretical_individual[0]
random_pool_df_wti_norm["1"] = reverse_theoretical_individual[1] - random_pool_df["1"].values + best_theoretical_individual[1]
random_pool_df_wti_norm["2"] = reverse_theoretical_individual[2] - random_pool_df["2"].values + best_theoretical_individual[2]
random_pool_df_wti_norm["3"] = reverse_theoretical_individual[3] - random_pool_df["3"].values + best_theoretical_individual[3]

random_pool_df_wti_norm["0"] = random_pool_df_wti_norm["0"] / reverse_theoretical_individual[0]
random_pool_df_wti_norm["1"] = random_pool_df_wti_norm["1"] / reverse_theoretical_individual[1]
random_pool_df_wti_norm["2"] = random_pool_df_wti_norm["2"] / reverse_theoretical_individual[2]
random_pool_df_wti_norm["3"] = random_pool_df_wti_norm["3"] / reverse_theoretical_individual[3]

print(random_pool_df_wti_norm.iloc[:, 1])
random_pool_dropped_wti_norm = random_pool_df_wti_norm.drop(columns=["Unnamed: 0"])
random_pool_scores_wti_norm = random_pool_dropped_wti_norm.to_numpy(dtype=np.float64)
random_pool_pareto_wti_norm = is_pareto(random_pool_scores_wti_norm, maximise=True)
random_pool_pareto_front_wti_norm = random_pool_scores_wti_norm[random_pool_pareto_wti_norm]
headers = ["0", "1", "2", "3"]
random_pool_pareto_front_df_wti_norm = pd.DataFrame(random_pool_pareto_front_wti_norm, columns=[headers])
random_pool_scores_wti_norm = random_pool_pareto_front_df_wti_norm.to_numpy(dtype=np.float64)

for experiment_name in experiment_list:
    # access externally
    filepath = "D:/Generator_RQ4/Systematic_Experiments/" + experiment_set_name + "BACKWARDS_" + experiment_name
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
                                           str(rep) + "_fits_final_gen_rerun.csv")  # load all of the generation score
            # print(results_path + origami_name + "_GA-experiment_rep_" +
            #                                str(rep) + "_scores_final_gen.csv")
            all_gen_scaffold_df = pd.read_csv(results_path + origami_name + "_GA-experiment_rep_" +
                                              str(rep) + "_scaffolds_final_gen_rerun.csv")

            # fill in blanks
            all_gen_score_df.fillna(0, inplace=True)

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

            # normalise results to between 0 and 1 using worst random pool values possible
            gen_scores["0"] = reverse_theoretical_individual[0] - gen_scores["0"] + best_theoretical_individual[0]
            gen_scores["1"] = reverse_theoretical_individual[1] - gen_scores["1"] + best_theoretical_individual[1]
            gen_scores["2"] = reverse_theoretical_individual[2] - gen_scores["2"] + best_theoretical_individual[2]
            gen_scores["3"] = reverse_theoretical_individual[3] - gen_scores["3"] + best_theoretical_individual[3]

            print(reverse_theoretical_individual)
            print(best_theoretical_individual)
            # normalise results to between 0 and 1 using worst random pool values possible
            gen_scores["0"] = gen_scores["0"] / reverse_theoretical_individual[0]
            gen_scores["1"] = gen_scores["1"] / reverse_theoretical_individual[1]
            gen_scores["2"] = gen_scores["2"] / reverse_theoretical_individual[2]
            gen_scores["3"] = gen_scores["3"] / reverse_theoretical_individual[3]
            scores = gen_scores.to_numpy(dtype=np.float64)

            print(scores[0])
            # apply function to gather pareto front scores
            pareto = is_pareto(scores, maximise=True)
            pareto_front = scores[pareto]
            pareto_front_df = pd.DataFrame(pareto_front)
            pareto_front_df.sort_values(0, inplace=True)
            pareto_front = pareto_front_df.values

            hv_calculated = hv.calc(scores)
            hyper_volumes.append([experiment_name, rep, hv_calculated])

    # create a df to analyse and get best/median, before adding best and median to total experiment df
    hyper_volume_df = pd.DataFrame(hyper_volumes)
    hyper_volume_df.columns = ["experiment", "rep", "hypervolume"]
    hyper_volume_df.sort_values(by=['hypervolume'], inplace=True, ascending=True)
    # best and med reps
    best_rep_df = hyper_volume_df.iloc[0, :].copy()
    med_rep_df = hyper_volume_df.iloc[median_value, :].copy()
    print(best_rep_df)
    print(med_rep_df)
    # all gens in loop
    best_rep_gen_score_df = pd.read_csv(results_path + origami_name + "_GA-experiment_rep_" +
                                        str(best_rep_df["rep"]) + "_fits_final_gen_rerun.csv")
    med_rep_gen_score_df = pd.read_csv(results_path + origami_name + "_GA-experiment_rep_" +
                                       str(med_rep_df["rep"]) + "_fits_final_gen_rerun.csv")

    # calculate number of generations from score
    generations = len(best_rep_gen_score_df) / individuals  # should be same for med
    # create generation column list for scores
    score_gen_list = []
    ind_counter = 0
    gen_counter = 0
    for scores_ in range(len(best_rep_gen_score_df)):
        ind_counter += 1
        score_gen_list.append(gen_counter)
        if ind_counter == individuals:
            ind_counter = 0
            gen_counter += 1
    final_gen = max(score_gen_list)

    # assign the generation column for filtering
    best_rep_gen_score_df['generation'] = score_gen_list
    med_rep_gen_score_df['generation'] = score_gen_list

    # create normalised scores of the pareto front / final gen
    best_rep_gen_scores = best_rep_gen_score_df.loc[best_rep_gen_score_df['generation'] == final_gen]\
        .drop(columns=['Unnamed: 0', 'generation'])
    med_rep_gen_scores = med_rep_gen_score_df.loc[med_rep_gen_score_df['generation'] == final_gen]\
        .drop(columns=['Unnamed: 0', 'generation'])

    best_rep_scores = best_rep_gen_scores.to_numpy(dtype=np.float64)
    med_rep_scores = med_rep_gen_scores.to_numpy(dtype=np.float64)

    # apply function to gather pareto front scores
    best_rep_pareto = is_pareto(best_rep_scores, maximise=True)
    best_rep_pareto_front = best_rep_scores[best_rep_pareto]
    best_rep_pareto_front_df = pd.DataFrame(best_rep_pareto_front)
    # best_rep_pareto_front_df.sort_values(0, inplace=True)
    best_rep_pareto_front = best_rep_pareto_front_df.values

    # apply to med rep
    med_rep_pareto = is_pareto(med_rep_scores, maximise=True)
    med_rep_pareto_front = med_rep_scores[med_rep_pareto]
    med_rep_pareto_front_df = pd.DataFrame(med_rep_pareto_front)
    # med_rep_pareto_front_df.sort_values(0, inplace=True)
    med_rep_pareto_front = med_rep_pareto_front_df.values

    # iterate over pareto front and calculate hyper-volumes
    best_rep_non_dom_len = len(best_rep_pareto_front)
    best_rep_inds_hv_list = []
    best_rep_inds = []
    # iterates over each individual on the pareto front, calculate hyper-volumes (hyper-lines, really)
    for j_best_inds in range(best_rep_non_dom_len):
        check_best_ind = best_rep_pareto_front[j_best_inds]
        check_best_ind_hv = hv.calc(check_best_ind)
        # append to lists
        best_rep_inds.append(check_best_ind)  # individual
        best_rep_inds_hv_list.append(check_best_ind_hv)  # individual's hyper-volume

    # why did I use a list instead of a data-frame
    best_hyper_volume_ind_df = pd.DataFrame(best_rep_inds_hv_list, columns=['individuals_hypervolume_values'])
    best_hyper_volume_ind_df["individual metrics"] = best_rep_inds
    best_hyper_volume_ind_df.reset_index(inplace=True)
    best_hyper_volume_ind_df = best_hyper_volume_ind_df.sort_values(by='individuals_hypervolume_values',
                                                                    ascending=False)

    # select the single best hyper-volume
    best_best_rep_details = best_hyper_volume_ind_df.iloc[0]
    best_best_rep_ind_hv = best_best_rep_details["individuals_hypervolume_values"]
    best_best_rep_ind = best_best_rep_details["individual metrics"]

    # why did I use a list instead of a data-frame
    # # get a sorted index list of the hyper-volume experiments, use idx 0 to get best
    check_best_sort_idx = sorted(range(len(best_rep_inds_hv_list)), key=best_rep_inds_hv_list.__getitem__, reverse=True)
    # apply the idx 0 to get the best individual on the pareto front
    list_best_best_rep_ind_hv = best_rep_inds_hv_list[check_best_sort_idx[0]]
    list_best_best_rep_ind = best_rep_pareto_front[check_best_sort_idx[0]]

    med_rep_non_dom_len = len(med_rep_pareto_front)
    med_rep_inds_hv_list = []
    med_rep_inds = []
    for j_med_inds in range(med_rep_non_dom_len):
        check_med_ind = med_rep_pareto_front[j_med_inds]
        check_med_ind_hv = hv.calc(check_med_ind)
        # append to lists
        med_rep_inds.append(check_med_ind)  # individual
        med_rep_inds_hv_list.append(check_med_ind_hv)  # individual's hyper-volume

    # why did I use a list instead of a data-frame
    med_hyper_volume_ind_df = pd.DataFrame(med_rep_inds_hv_list, columns=['individuals_hypervolume_values'])
    med_hyper_volume_ind_df["individual metrics"] = med_rep_inds
    med_hyper_volume_ind_df.reset_index(inplace=True)
    med_hyper_volume_ind_df = med_hyper_volume_ind_df.sort_values(by='individuals_hypervolume_values', ascending=False)

    # select the single best hyper-volume
    best_med_rep_details = med_hyper_volume_ind_df.iloc[0]
    best_med_rep_ind_hv = best_med_rep_details["individuals_hypervolume_values"]
    best_med_rep_ind = best_med_rep_details["individual metrics"]

    best_rep_decision = tp.topsis(best_rep_pareto_front, weights, costs)
    med_rep_decision = tp.topsis(med_rep_pareto_front, weights, costs)

    best_rep_topsis = best_rep_pareto_front[best_rep_decision[0]]
    # best_rep_topsis = str(best_rep_decision).split(":")[1][1:].replace(" ", ",").strip("[").strip("]").split(",")
    # while '' in best_rep_topsis:
    #     best_rep_topsis.remove('')
    # best_rep_topsis_list = [int(i) if i.isdigit() else float(i) for i in best_rep_topsis]
    print("best rep topsis", best_rep_topsis)

    med_rep_topsis = med_rep_pareto_front[med_rep_decision[0]]
    # med_rep_topsis = str(med_rep_decision).split(":")[1][1:].replace(" ", ",").strip("[").strip("]").split(",")
    # while '' in med_rep_topsis:
    #     med_rep_topsis.remove('')
    # med_rep_topsis_list = [int(i) if i.isdigit() else float(i) for i in med_rep_topsis]
    print("med rep topsis", med_rep_topsis)

    # add the data to the data frames
    best_rep_df["best individual metrics via hypervolume Metric 1"] = best_best_rep_ind[0].round(3)
    best_rep_df["best individual metrics via hypervolume Metric 2"] = best_best_rep_ind[1].round(3)
    best_rep_df["best individual metrics via hypervolume Metric 3"] = best_best_rep_ind[2].round(3)
    best_rep_df["best individual metrics via hypervolume Metric 4"] = best_best_rep_ind[3].round(3)

    best_rep_df["best individual metrics via topsis Metric 1"] = best_rep_topsis[0].round(3)
    best_rep_df["best individual metrics via topsis Metric 2"] = best_rep_topsis[1].round(3)
    best_rep_df["best individual metrics via topsis Metric 3"] = best_rep_topsis[2].round(3)
    best_rep_df["best individual metrics via topsis Metric 4"] = best_rep_topsis[3].round(3)

    best_rep_df["best individual hypervolume"] = best_best_rep_ind_hv
    best_rep_df["pareto front length"] = best_rep_non_dom_len
    best_rep_df["repetition_type"] = "best hypervolume"

    med_rep_df["best individual metrics via hypervolume Metric 1"] = best_med_rep_ind[0].round(3)
    med_rep_df["best individual metrics via hypervolume Metric 2"] = best_med_rep_ind[1].round(3)
    med_rep_df["best individual metrics via hypervolume Metric 3"] = best_med_rep_ind[2].round(3)
    med_rep_df["best individual metrics via hypervolume Metric 4"] = best_med_rep_ind[3].round(3)

    med_rep_df["best individual metrics via topsis Metric 1"] = med_rep_topsis[0].round(3)
    med_rep_df["best individual metrics via topsis Metric 2"] = med_rep_topsis[1].round(3)
    med_rep_df["best individual metrics via topsis Metric 3"] = med_rep_topsis[2].round(3)
    med_rep_df["best individual metrics via topsis Metric 4"] = med_rep_topsis[3].round(3)

    med_rep_df["best individual hypervolume"] = best_med_rep_ind_hv
    med_rep_df["pareto front length"] = med_rep_non_dom_len
    med_rep_df["repetition_type"] = "median hypervolume"

    combined_rep_df = pd.concat([best_rep_df, med_rep_df], axis=1)
    # transpose the data frame to show all experiments per row
    combined_rep_df = combined_rep_df.transpose()
    # print(combined_rep_df.columns)
    # add to all outer experiments
    all_hyper_volumes_df = pd.concat([all_hyper_volumes_df, combined_rep_df])

    # #### ADDED GRA METRIC #####
    # best and med reps GRA
    best_rep_df_gra = hyper_volume_df.iloc[0, :].copy()
    med_rep_df_gra = hyper_volume_df.iloc[median_value, :].copy()
    # calculate the GRA metric and create plots for this
    import matplotlib.pyplot as plt
    import seaborn as sns
    experiment_name_for_plot = experiment_name.strip("CXPB")
    experiment_name_for_plot = experiment_name_for_plot.strip("_SWEEP_") + " repetition " + str(best_rep_df["rep"])
    # Establish a gray correlation model, standardize data
    best_rep_gra_model = GrayRelationalCoefficient(best_rep_pareto_front, standard=False)
    best_rep_cors = best_rep_gra_model.get_calculate_relational_coefficient()

    # create data frame to sort and index the values obtained
    best_rep_gra_df = pd.DataFrame()

    best_rep_max_corr = max(best_rep_cors, key=lambda x: x.tolist())
    # retrieve index of best
    best_rep_max_index_ndarray = np.argwhere(best_rep_cors == best_rep_max_corr)
    best_rep_max_index_list_ndarray = best_rep_max_index_ndarray.tolist()
    best_rep_max_index_list = [item[0] for item in best_rep_max_index_list_ndarray]
    best_rep_max_index_duplicate = set([x for x in best_rep_max_index_list if best_rep_max_index_list.count(x) > 1])
    best_rep_max_index = best_rep_max_index_duplicate.pop()
    best_rep_best_gra_ind = best_rep_pareto_front[best_rep_max_index]

    # apply for the med_rep
    med_rep_gra_model = GrayRelationalCoefficient(med_rep_pareto_front, standard=False)
    med_rep_cors = med_rep_gra_model.get_calculate_relational_coefficient()
    med_rep_max_corr = max(med_rep_cors, key=lambda x: x.tolist())
    # retrieve index of best
    med_rep_max_index_ndarray = np.argwhere(med_rep_cors == med_rep_max_corr)
    med_rep_max_index_list_ndarray = med_rep_max_index_ndarray.tolist()
    med_rep_max_index_list = [item[0] for item in med_rep_max_index_list_ndarray]
    med_rep_max_index_duplicate = set([x for x in med_rep_max_index_list if med_rep_max_index_list.count(x) > 1])
    med_rep_max_index = med_rep_max_index_duplicate.pop()
    med_rep_best_gra_ind = med_rep_pareto_front[med_rep_max_index]

    # add the data to the data frames
    best_rep_df_gra["best individual GRA Correlations"] = best_rep_max_corr
    best_rep_df_gra["best individual metrics via GRA Metric 1"] = best_rep_best_gra_ind[0].round(3)
    best_rep_df_gra["best individual metrics via GRA Metric 2"] = best_rep_best_gra_ind[1].round(3)
    best_rep_df_gra["best individual metrics via GRA Metric 3"] = best_rep_best_gra_ind[2].round(3)
    best_rep_df_gra["best individual metrics via GRA Metric 4"] = best_rep_best_gra_ind[3].round(3)
    best_rep_df_gra["repetition_type"] = "best hypervolume"

    med_rep_df_gra["best individual GRA Correlations"] = med_rep_max_corr
    med_rep_df_gra["best individual metrics via GRA Metric 1"] = med_rep_best_gra_ind[0].round(3)
    med_rep_df_gra["best individual metrics via GRA Metric 2"] = med_rep_best_gra_ind[1].round(3)
    med_rep_df_gra["best individual metrics via GRA Metric 3"] = med_rep_best_gra_ind[2].round(3)
    med_rep_df_gra["best individual metrics via GRA Metric 4"] = med_rep_best_gra_ind[3].round(3)
    med_rep_df_gra["repetition_type"] = "median hypervolume"

    combined_gra_rep_df = pd.concat([best_rep_df_gra, med_rep_df_gra], axis=1)
    # transpose the data frame to show all experiments per row
    combined_gra_rep_df = combined_gra_rep_df.transpose()
    # add to all outer experiments
    all_gra_df = pd.concat([all_gra_df, combined_gra_rep_df])

print(all_hyper_volumes_df.columns)
print(all_gra_df.columns)

# store as csv table
total_df = pd.merge(all_hyper_volumes_df, all_gra_df, how="inner")
total_df = total_df.round(decimals=0)
total_df.to_csv("Analysis_Output/" + experiment_set_name.strip("/") + "reverse_non_norm_total_table.csv")
# all_hyper_volumes_df.to_csv("Analysis_Output/" + experiment_set_name.strip("/") + "_hypervolume_table.csv")
# all_gra_df.to_csv("Analysis_Output/" + experiment_set_name.strip("/") + "_gra_table.csv")

# calculate hyper-volume
print("random_pool hyper-volume:", hv.calc(random_pool_scores_wti_norm))
random_pool_calculated = hv.calc(random_pool_scores_wti_norm)
