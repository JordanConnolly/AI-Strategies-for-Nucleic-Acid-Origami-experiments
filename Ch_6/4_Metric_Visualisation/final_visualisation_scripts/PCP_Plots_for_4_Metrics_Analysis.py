from pymoo.factory import get_problem, get_reference_directions
from pymoo.problems.many.dtlz import DTLZ1
from pymoo.visualization.pcp import PCP
from pymoo.factory import get_performance_indicator
import pandas as pd
import numpy as np
from os import path
from textwrap import wrap


"""
SOURCE: https://pymoo.org/visualization/pcp.html
"""

# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # SET EXPERIMENT PARAMETERS # # # # #

# create list of experiment names
experiment_list = [
                   "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_10",
                   "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_25",
                   "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_50",
                   ]

list_of_origami_names = ['ball', '6hb', 'dunn', 'DBS_square',
                         'minitri', 'Abrick', 'fourfinger-circular',
                         'fourfinger-linear',
                         'nanoribbonRNA', 'hj']


for origami_name in list_of_origami_names:
    experiment_set_name = origami_name + "_SWEEP/"
    # create variable for num of individual in gen
    individuals = 40


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
        filepath = "E:/PhD Files/RQ4/4_Metrics_SWEEP_Final_Energy_Model/" + experiment_set_name + experiment_name
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
                # print(results_path + origami_name + "_GA-experiment_rep_" +
                #                                str(rep) + "_scores_final_gen.csv")
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
                gen_scores = all_gen_score_df.loc[
                    all_gen_score_df['generation'] == final_gen].drop(columns=['Unnamed: 0',
                                                                               'generation'])

                scores = gen_scores.to_numpy(dtype=np.float64)
                # apply function to gather pareto front scores
                pareto = is_pareto(scores)
                pareto_front = scores[pareto]
                pareto_front_df = pd.DataFrame(pareto_front)
                pareto_front_df.sort_values(0, inplace=True)
                pareto_front = pareto_front_df.values

                # create lower and upper bounds for all of the metrics
                metric_1_upper_bound = max(pareto_front[0])  # find UPPER BOUND
                metric_1_upper_bound = metric_1_upper_bound + (metric_1_upper_bound / 100 * 2)
                metric_1_upper_bound -= metric_1_upper_bound % +100  # create ceiling
                metric_1_upper_bound = int(np.round(metric_1_upper_bound, 0))

                # create lower and upper bounds for all of the metrics
                metric_2_upper_bound = max(pareto_front[1])  # find UPPER BOUND
                metric_2_upper_bound = metric_2_upper_bound + (metric_2_upper_bound / 100 * 2)
                metric_2_upper_bound -= metric_2_upper_bound % +100  # create ceiling
                metric_2_upper_bound = int(np.round(metric_2_upper_bound, 0))

                # create lower and upper bounds for all of the metrics
                metric_3_upper_bound = max(pareto_front[2])  # find UPPER BOUND
                metric_3_upper_bound = metric_3_upper_bound + (metric_3_upper_bound / 100 * 2)
                metric_3_upper_bound -= metric_3_upper_bound % +0.1  # create ceiling
                metric_3_upper_bound = int(np.round(metric_3_upper_bound, 0))

                # create lower and upper bounds for all of the metrics
                metric_4_upper_bound = max(pareto_front[3])  # find UPPER BOUND
                metric_4_upper_bound = metric_4_upper_bound + (metric_4_upper_bound / 100 * 2)
                metric_4_upper_bound -= metric_4_upper_bound % +0.1  # create ceiling
                metric_4_upper_bound = int(np.round(metric_4_upper_bound, 0))

                # pareto front
                plot = PCP()
                plot.set_axis_style(color="grey", alpha=0.5)
                plot.add(pareto_front, color="grey", alpha=0.4)
                # plot.add(pareto_front[15], linewidth=5, color="red")  # you can also add individuals
                # plot.add(pareto_front[15], linewidth=5, color="red")  # add random values
                plot.bounds = [[0, 0, 0, 0], [metric_1_upper_bound, metric_2_upper_bound,
                                              metric_3_upper_bound, metric_4_upper_bound]]
                # fix the plot bounds
                plot.normalize_each_axis = False
                plot.show()


    # plot = PCP()
    # plot.set_axis_style(color="grey", alpha=0.5)
    # plot.add(F, color="grey", alpha=0.3)
    # plot.add(F[15], linewidth=5, color="red")
    # plot.add(F[32], linewidth=5, color="blue")
    # plot.show()
    # plot = PCP(title=("Run", {'pad': 30}),
    #            n_ticks=10,
    #            legend=(True, {'loc': "upper left"}),
    #            labels=["Metric 1", "Metric 2", "Metric 3", "Metric 4"]
    #            )
    # plot.set_axis_style(color="grey", alpha=1)
    # plot.add(F, color="grey", alpha=0.3)
    # plot.add(F[15], linewidth=5, color="red", label="Solution A")
    # plot.add(F[32], linewidth=5, color="blue", label="Solution B")
    # plot.show()
    # plot.reset()
    # plot.normalize_each_axis = False
    # plot.bounds = [[1, 1, 1, 2], [32, 32, 32, 32]]
    # plot.show()
