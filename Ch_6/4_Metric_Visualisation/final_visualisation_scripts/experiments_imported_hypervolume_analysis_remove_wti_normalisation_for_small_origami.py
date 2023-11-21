from pymoo.factory import get_performance_indicator
import pandas as pd
import numpy as np
from os import path
from textwrap import wrap
# -*- coding: utf-8 -*-
from GRA_metric import GrayRelationalCoefficient, plot_average_grey_relational_coefficient
from topsis import topsis
import topsispy as tp  # implementation that is used, doesn't matter which is used

# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # SET EXPERIMENT PARAMETERS # # # # #

# create list of experiment names
experiment_list = [
                   "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_10",
                   "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_25",
                   "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_50"
                   ]

origami_name = "hj"
experiment_set_name = origami_name + "_SWEEP/"
# create variable for num of individual in gen
individuals = 40
median = 4

worst_theoretical_individual_df = pd.read_csv("worst_theoretical_individual/" +
                                              origami_name + "_Worst_Theoretical_Individual.csv")
worst_theoretical_individual_dropped_df = worst_theoretical_individual_df.drop(columns="Unnamed: 0")
worst_theoretical_individual = worst_theoretical_individual_dropped_df.values

# create performance indicator for use later
hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2, 1.2, 1.2]))
wti_hv = get_performance_indicator("hv", ref_point=np.array([worst_theoretical_individual[0][0],
                                                             worst_theoretical_individual[1][0],
                                                             worst_theoretical_individual[2][0],
                                                             worst_theoretical_individual[3][0]]))

# calculate the best alternative using TOPSIS implementation
weights = [0.25, 0.25, 0.25, 0.25]
costs = [-1, -1, -1, -1]

# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # #

# create a data frame to store hyper-volume analysis results for all experiments in list
all_hyper_volumes_df = pd.DataFrame()
# create a data frame to store GRA (extra metric)
all_gra_df = pd.DataFrame()


def is_pareto(costs, maximise=False):
    """
    :param costs: An (n_points, n_costs) array
    :param maximise: boolean. True for maximising, False for minimising
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)  # Remove dominated points
                is_efficient[i] = True
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Remove dominated points
                is_efficient[i] = True
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


for experiment_name in experiment_list:
    # access externally
    filepath = "E:/PhD Files/RQ4/4_Metrics_SWEEP_Final_Energy_Model/" + experiment_set_name + experiment_name
    input_path = filepath  # find the path for the input data
    print("\n", input_path)
    metadata_path = input_path + "/metadata/"  # specify the metadata path
    results_path = input_path + "/results/"  # specify the results path
    plot_path = input_path + "/plots/"  # specify the plots path
    hyper_volumes = []

    for i in range(1, 11, 1):
        rep = i
        if rep != 35:
            # all gens in loop
            all_gen_score_df = pd.read_csv(results_path + origami_name + "_GA-experiment_rep_" +
                                           str(rep) + "_scores_final_gen.csv")  # load all of the generation score
            # print(results_path + origami_name + "_GA-experiment_rep_" +
            #                                str(rep) + "_scores_final_gen.csv")
            all_gen_scaffold_df = pd.read_csv(results_path + origami_name + "_GA-experiment_rep_" +
                                              str(rep) + "_scaffolds_final_gen.csv")

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
            gen_scores["0"] = gen_scores["0"] / worst_theoretical_individual[0]
            gen_scores["1"] = gen_scores["1"] / worst_theoretical_individual[1]
            gen_scores["2"] = gen_scores["2"] / worst_theoretical_individual[2]
            gen_scores["3"] = gen_scores["3"] / worst_theoretical_individual[3]
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
    # best and med reps
    best_rep_df = hyper_volume_df.iloc[0, :].copy()
    med_rep_df = hyper_volume_df.iloc[median, :].copy()
    # all gens in loop
    best_rep_gen_score_df = pd.read_csv(results_path + origami_name + "_GA-experiment_rep_" +
                                        str(best_rep_df["rep"]) + "_scores_final_gen.csv")
    med_rep_gen_score_df = pd.read_csv(results_path + origami_name + "_GA-experiment_rep_" +
                                       str(med_rep_df["rep"]) + "_scores_final_gen.csv")

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
    best_rep_pareto = is_pareto(best_rep_scores)
    best_rep_pareto_front = best_rep_scores[best_rep_pareto]
    best_rep_pareto_front_df = pd.DataFrame(best_rep_pareto_front)
    # best_rep_pareto_front_df.sort_values(0, inplace=True)
    best_rep_pareto_front = best_rep_pareto_front_df.values

    # apply to med rep
    med_rep_pareto = is_pareto(med_rep_scores)
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
        check_best_ind_hv = wti_hv.calc(check_best_ind)
        # append to lists
        best_rep_inds.append(check_best_ind)  # individual
        best_rep_inds_hv_list.append(check_best_ind_hv)  # individual's hyper-volume

    # why did I use a list instead of a data-frame
    best_hyper_volume_ind_df = pd.DataFrame(best_rep_inds_hv_list, columns=['individuals_hypervolume_values'])
    best_hyper_volume_ind_df["individual metrics"] = best_rep_inds
    best_hyper_volume_ind_df.reset_index(inplace=True)
    best_hyper_volume_ind_df = best_hyper_volume_ind_df.sort_values(by='individuals_hypervolume_values', ascending=False)

    # select the single best hyper-volume
    print(best_hyper_volume_ind_df)
    best_best_rep_details = best_hyper_volume_ind_df.iloc[0]
    best_best_rep_ind_hv = best_best_rep_details["individuals_hypervolume_values"]
    best_best_rep_ind = best_best_rep_details["individual metrics"]
    print(best_best_rep_ind)

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
        check_med_ind_hv = wti_hv.calc(check_med_ind)
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
    print(best_med_rep_details)
    best_med_rep_ind_hv = best_med_rep_details["individuals_hypervolume_values"]
    best_med_rep_ind = best_med_rep_details["individual metrics"]

    best_rep_decision = tp.topsis(best_rep_pareto_front, weights, costs)
    med_rep_decision = tp.topsis(med_rep_pareto_front, weights, costs)

    print("best rep:", best_rep_df["rep"])
    best_rep_topsis = best_rep_pareto_front[best_rep_decision[0]]
    # best_rep_topsis = str(best_rep_decision).split(":")[1][1:].replace(" ", ",").strip("[").strip("]").split(",")
    # while '' in best_rep_topsis:
    #     best_rep_topsis.remove('')
    # best_rep_topsis_list = [int(i) if i.isdigit() else float(i) for i in best_rep_topsis]
    print("best rep topsis", best_rep_topsis)

    print("med rep:", med_rep_df["rep"])
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

print(all_hyper_volumes_df.columns)

# store as csv table
total_df = all_hyper_volumes_df
# total_df = total_df.round(decimals=0)
total_df.to_csv("Analysis_Output/" + experiment_set_name.strip("/") + "_non_norm_total_table.csv")
# all_hyper_volumes_df.to_csv("Analysis_Output/" + experiment_set_name.strip("/") + "_hypervolume_table.csv")
# all_gra_df.to_csv("Analysis_Output/" + experiment_set_name.strip("/") + "_gra_table.csv")
