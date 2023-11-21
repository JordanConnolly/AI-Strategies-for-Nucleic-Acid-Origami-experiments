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
import re
import glob
from pymoo.factory import get_performance_indicator
from textwrap import wrap
# -*- coding: utf-8 -*-
# from GRA_metric import GrayRelationalCoefficient, plot_average_grey_relational_coefficient
# from topsis import topsis
# from topsis import topsis
import topsispy as tp  # implementation that is used, doesn't matter which is used


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
experiment_list = [
    "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_10",
    "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_25",
    "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_50",
]

list_of_origami_names = ["hj", "nanoribbonRNA", "fourfinger-linear",
                         "fourfinger-circular", "minitri", "DBS_square",
                         "Abrick", "6hb", "ball", "dunn"]

list_of_origami_names_for_plot = ["Single HJ",
                                  "Nanoribbon (RNA)",
                                  "M1.3 Four Finger (Linear)",
                                  "M1.3 Four Finger (Circular)",
                                  "Mini Triangle",
                                  "Jurek Square",
                                  "Solid Brick",
                                  "6 Helix Bundle",
                                  "Ball",
                                  "Dannenberg Tile with TTTT overhangs"]

# create list of experiment names
order_permutation_list = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],
                          [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3],
                          [0, 1, 2, 3], [0, 1, 2, 3]]


for i in range(len(list_of_origami_names)):

    # set the origami for plotting
    origami_name = list_of_origami_names[i]
    origami_name_for_plot = list_of_origami_names_for_plot[i]
    order_permutation = order_permutation_list[i]

    upper_bound_percentage = 5
    lower_bound_percentage = 5

    experiment_set_name = origami_name + "_SWEEP/"

    # create variable for num of individual in gen
    individuals = 40
    median_value = 4
    # calculate the best alternative using TOPSIS implementation
    weights = [0.25, 0.25, 0.25, 0.25]
    costs = [-1, -1, -1, -1]

    # access and import the random search scores
    random_scores_path = "E:/PhD Files/RQ4/4_Metrics_SWEEP_Final_Energy_Model/" + \
                         origami_name + "_SWEEP/Random_Walk_300000/metadata/"
    # random_scores_path = "D:/Generator_RQ4/Systematic_Experiments/" + origami_name + "_SWEEP/Random_Walk_300000/metadata/"
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
    new_worst_random_pool = []
    new_worst_random_pool.append(max(random_pool_df.iloc[:, 1].values))
    new_worst_random_pool.append(max(random_pool_df.iloc[:, 2].values))
    new_worst_random_pool.append(max(random_pool_df.iloc[:, 3].values))
    new_worst_random_pool.append(max(random_pool_df.iloc[:, 4].values))
    print(new_worst_random_pool)
    worst_random_pool_df = pd.DataFrame(new_worst_random_pool)
    worst_theoretical_individual = new_worst_random_pool

    # create performance indicator for use later
    hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2, 1.2, 1.2]))
    wti_hv = get_performance_indicator("hv", ref_point=np.array([worst_theoretical_individual[0],
                                                                 worst_theoretical_individual[1],
                                                                 worst_theoretical_individual[2],
                                                                 worst_theoretical_individual[3]]))

    """
    SOURCE: https://pymoo.org/visualization/pcp.html
    """


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


    # apply function to gather pareto front scores
    random_pool_dropped = random_pool_df.drop(columns=["Unnamed: 0"])
    random_pool_scores = random_pool_dropped.to_numpy(dtype=np.float64)
    random_pool_pareto = is_pareto(random_pool_scores)
    random_pool_pareto_front = random_pool_scores[random_pool_pareto]
    random_pool_pareto_front_df = pd.DataFrame(random_pool_pareto_front)
    random_pool_pareto_front = random_pool_pareto_front_df.values

    random_pool_decision = tp.topsis(random_pool_pareto_front, weights, costs)
    print(random_pool_decision)

    experiment_set_count = 0
    for experiment_name in experiment_list:
        experiment_set_count += 1
        # access externally
        filepath = "E:/PhD Files/RQ4/4_Metrics_SWEEP_Final_Energy_Model/" + experiment_set_name + experiment_name
        # filepath = "D:/Generator_RQ4/Systematic_Experiments/" + experiment_set_name + experiment_name
        input_path = filepath  # find the path for the input data
        print("\n", input_path)
        metadata_path = input_path + "/metadata/"  # specify the metadata path
        results_path = input_path + "/results/"  # specify the results path
        plot_path = input_path + "/plots/"  # specify the plots path
        hyper_volumes = []

        for rep in range(1, 11, 1):
            if rep != 35:
                # all gens in loop
                # all_gen_score_df = pd.read_csv(results_path + origami_name + "_GA-experiment_rep_" +
                #                                str(rep) + "_scores_final_gen.csv")  # load all of the generation score
                # print(results_path + origami_name + "_GA-experiment_rep_" +
                #                                str(rep) + "_scores_final_gen.csv")
                all_gen_scaffold_df = pd.read_csv(results_path + origami_name + "_GA-experiment_rep_" +
                                                  str(rep) + "_scaffolds_final_gen.csv")
                # print all gens
                # # print(all_gen_scaffold_df.iloc[0])
                # # print(all_gen_score_df.iloc[0])
                # # calculate number of generations from score
                # generations = len(all_gen_score_df) / individuals
                #
                # # create generation column list for scores
                # score_gen_list = []
                # ind_counter = 0
                # gen_counter = 0
                # for k in range(len(all_gen_score_df)):
                #     ind_counter += 1
                #     score_gen_list.append(gen_counter)
                #     if ind_counter == individuals:
                #         ind_counter = 0
                #         gen_counter += 1
                #
                # final_gen = max(score_gen_list)
                # # assign the generation column for filtering
                # all_gen_score_df['generation'] = score_gen_list
                # # print(all_gen_score_df)
                #
                # # create normalised scores of the pareto front / final gen
                # gen_scores = all_gen_score_df.loc[all_gen_score_df['generation'] == final_gen].drop(columns=['Unnamed: 0',
                #                                                                                              'generation'
                #                                                                                              ])
                gen_scores = pd.read_csv(results_path + origami_name + "_GA-experiment_rep_" +
                                         str(rep) + "_fits_gen_250.csv").drop(columns=["Unnamed: 0"])
                print(gen_scores.columns)
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
                hv_calculated = hv.do(scores)
                print(hv_calculated)
                hyper_volumes.append([experiment_name, rep, hv_calculated])

        # create a df to analyse and get best/median, before adding best and median to total experiment df
        hyper_volume_df = pd.DataFrame(hyper_volumes)
        hyper_volume_df.columns = ["experiment", "rep", "hypervolume"]
        hyper_volume_df.sort_values(by=['hypervolume'], inplace=True, ascending=False)
        # best and med reps
        best_rep_df = hyper_volume_df.iloc[0, :].copy()
        med_rep_df = hyper_volume_df.iloc[median_value, :].copy()
        print(str(best_rep_df['rep']))
        print(str(med_rep_df['rep']))
        # all gens in loop
        best_rep_gen_score_df = pd.read_csv(results_path + origami_name + "_GA-experiment_rep_" +
                                            str(best_rep_df["rep"]) + "_fits_gen_250.csv").drop(columns=["Unnamed: 0"])
        med_rep_gen_score_df = pd.read_csv(results_path + origami_name + "_GA-experiment_rep_" +
                                           str(med_rep_df["rep"]) + "_fits_gen_250.csv").drop(columns=["Unnamed: 0"])
        #
        # # calculate number of generations from score
        # generations = len(best_rep_gen_score_df) / individuals  # should be same for med
        # # create generation column list for scores
        # score_gen_list = []
        # ind_counter = 0
        # gen_counter = 0
        # for scores_ in range(len(best_rep_gen_score_df)):
        #     ind_counter += 1
        #     score_gen_list.append(gen_counter)
        #     if ind_counter == individuals:
        #         ind_counter = 0
        #         gen_counter += 1
        # final_gen = max(score_gen_list)
        #
        # # assign the generation column for filtering
        # best_rep_gen_score_df['generation'] = score_gen_list
        # med_rep_gen_score_df['generation'] = score_gen_list
        #
        # # create normalised scores of the pareto front / final gen
        # best_rep_gen_scores = best_rep_gen_score_df.loc[best_rep_gen_score_df['generation'] == final_gen] \
        #     .drop(columns=['Unnamed: 0', 'generation'])
        # med_rep_gen_scores = med_rep_gen_score_df.loc[med_rep_gen_score_df['generation'] == final_gen] \
        #     .drop(columns=['Unnamed: 0', 'generation'])

        best_rep_scores = best_rep_gen_score_df.to_numpy(dtype=np.float64)
        med_rep_scores = med_rep_gen_score_df.to_numpy(dtype=np.float64)

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

        med_rep_decision = tp.topsis(med_rep_pareto_front, weights, costs)
        print("med rep:", med_rep_df["rep"])
        med_rep_topsis = med_rep_pareto_front[med_rep_decision[0]]

        topsis_selected_individual = med_rep_topsis
        print(med_rep_topsis)
        topsis_selected_random_pareto_front_individual = random_pool_pareto_front[random_pool_decision[0]]

        # add an experiment name for plot
        experiment_name_for_plot = experiment_name.strip("CXPB_SWEEP")

        # re-order columns
        tsi_cols = topsis_selected_individual
        # re-order columns
        tsrpfi_cols = topsis_selected_random_pareto_front_individual

        tsi_cols = tsi_cols[order_permutation]
        tsrpfi_cols = tsrpfi_cols[order_permutation]


        def create_metric_list(individual):
            individual_metrics = []
            counter = 0
            for i in range(len(individual)):
                if counter == i:
                    add_to_list = individual[i]
                    individual_metrics.append(add_to_list)
                counter += 1
            return individual_metrics


        # create upper and lower bounds
        lower_bound_list = []
        upper_bound_list = []

        # iterate over the number of metrics
        for selected_metric_index in range(len(med_rep_topsis)):
            check_med_metric = med_rep_topsis[selected_metric_index]
            check_selector_metric = topsis_selected_random_pareto_front_individual[selected_metric_index]

            lower_bound_metric = 0
            upper_bound_metric = 0
            if check_med_metric < check_selector_metric:
                lower_bound_metric = check_med_metric
                upper_bound_metric = check_selector_metric
            else:
                lower_bound_metric = check_selector_metric
                upper_bound_metric = check_med_metric
            percentage_to_decrease = lower_bound_metric / 100 * lower_bound_percentage
            lower_bound_of_metric = lower_bound_metric - percentage_to_decrease
            if lower_bound_of_metric <= 0:
                lower_bound_of_metric = 0
            lower_bound_list.append(lower_bound_of_metric)

            percentage_to_increase = upper_bound_metric / 100 * upper_bound_percentage
            upper_bound_of_metric = upper_bound_metric + percentage_to_increase
            upper_bound_list.append(upper_bound_of_metric)

        lower_bounds = np.array(lower_bound_list)
        upper_bounds = np.array(upper_bound_list)

        lower_bounds = lower_bounds[order_permutation]
        upper_bounds = upper_bounds[order_permutation]

        title = ("\n".join(wrap("Median Experiment from " + origami_name_for_plot + " " +
                                "Genetic Algorithm Experiment Set " + str(experiment_set_count), 55)))
        # pareto front
        plot = PCP(title=(title, {'pad': 30}),
                   n_ticks=10,
                   legend=(True, {'loc': "upper right"}),
                   labels=["Metric " + str(order_permutation[0]+1),
                           "Metric " + str(order_permutation[1]+1),
                           "Metric " + str(order_permutation[2]+1),
                           "Metric " + str(order_permutation[3]+1)])
        plot.set_axis_style(color="grey", alpha=0.8)
        # plot.add(med_rep_pareto_front, color="grey", alpha=0.4)
        # plot.add(pareto_front[15], linewidth=5, color="red")  # you can also add individuals
        plot.add(tsrpfi_cols, color="red", alpha=0.8,
                 label="TOPSIS selected Individual - 100,000 Scaffold Pareto Front from Selector")  # add random values
        plot.add(tsi_cols, color="blue", alpha=0.8,
                 label="TOPSIS selected Individual - Median Repetition Forward Problem GA")  # add random values
        plot.bounds = [lower_bounds, upper_bounds]
        plot.normalize_each_axis = False

        plot.save(basepath + "/Plot_Output/" + "PCP_plot_selection_with_random_55c_4_Metric_" + origami_name_for_plot +
                  "_genetic_algorithm_experiment_set_" + str(experiment_set_count) + ".png",
                  bbox_inches='tight', format='png')
        plot.show()
