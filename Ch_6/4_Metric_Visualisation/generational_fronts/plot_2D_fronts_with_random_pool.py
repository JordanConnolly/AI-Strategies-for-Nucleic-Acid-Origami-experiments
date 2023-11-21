import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import time
from os import path
import re
import glob
from textwrap import wrap

# base_path / path to all files in this project
basepath = path.dirname(__file__)

# access and import the random search scores
rp_path = path.abspath(path.join(basepath, "..", "Pareto_Plots", "random_search_results", "DBS_square",
                                 "DBS_square_RW-experiment_TESTrandom_walk_scaffold_scores.csv"))
random_pool_df = pd.read_csv(rp_path)

# access and import the details for the plot from a table
filepath = path.abspath(path.join(basepath, "..", "GA_Raw_Results_Analysis", "Analysis_Output", "CXPB_SWEEP_SET_2_hypervolume_table_select.csv"))
table_df = pd.read_csv(filepath)


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


random_pool_dropped = random_pool_df.drop(columns=["Unnamed: 0"])
# calculate pareto scores
random_pool_scores = random_pool_dropped.to_numpy(dtype=np.float64)
random_pool_pareto = is_pareto(random_pool_scores)
random_pool_pareto_front = random_pool_scores[random_pool_pareto]
headers = ["0", "1", "2"]
random_pool_pareto_front_df = pd.DataFrame(random_pool_pareto_front, columns=[headers])
random_pool_x_pareto = random_pool_pareto_front_df.iloc[:, 0]
random_pool_y_pareto = random_pool_pareto_front_df.iloc[:, 1]
random_pool_z_pareto = random_pool_pareto_front_df.iloc[:, 2]


# experiment_list = [17, 5]

# for experiment in experiment_list:
for experiment in table_df.itertuples():

    rep = experiment[3]
    experiment_name = experiment[2]

    if rep > 0:
        # name of experiment file
        # experiment_name = "CXPB_SWEEP_CXPB_90_MUTATIONS_16BP_INDMUTATION_50"
        # access externally
        filepath = "D:/Generator_RQ4/Systematic_Experiments/CXPB_SWEEP_SET_2/" + experiment_name
        input_path = filepath  # find the path for the input data
        metadata_path = input_path + "/metadata/"  # specify the metadata path
        results_path = input_path + "/results/"  # specify the results path
        plot_path = input_path + "/plots/"  # specify the plots path
        origami_name = "DBS_square"

        # set display precision
        pd.set_option("display.precision", 20)
        np.set_printoptions(precision=40, formatter='longfloat')

        # import the paths and name the origami of interest
        # rep = experiment
        rep_name = "rep_" + str(rep)
        print(experiment_name + rep_name)
        # you may as well not use the metadata path, and use the compiled results pathways
        gen0_scaffold_df = pd.read_csv(metadata_path + origami_name + "_GA-experiment_" + rep_name + "_scaffolds_gen_0.csv")

        # all gens in loop
        all_gen_score_df = pd.read_csv(results_path + origami_name + "_GA-experiment_" + rep_name + "_scores_final_gen.csv")  # load all of the generation score
        all_gen_scaffold_df = pd.read_csv(results_path + origami_name + "_GA-experiment_" + rep_name + "_scaffolds_final_gen.csv")

        # create variable for num of individual in gen
        individuals = 100
        generations = len(all_gen_scaffold_df) / individuals

        # create generation column list for scores
        score_gen_list = []
        ind_counter = 0
        gen_counter = 0
        for i in range(len(all_gen_score_df)):
            ind_counter += 1
            score_gen_list.append(gen_counter)
            if ind_counter == individuals:
                ind_counter = 0
                gen_counter += 1

        final_gen = max(score_gen_list)
        # assign the generation column for filtering
        all_gen_score_df['generation'] = score_gen_list
        final_gen_scores = all_gen_score_df.loc[all_gen_score_df['generation'] == final_gen].drop(columns=['generation'])

        experiment_name_for_plot = experiment_name.strip("CXPB")
        experiment_name_for_plot = experiment_name_for_plot.strip("_SWEEP_")
        for i in range(final_gen + 1):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # create pareto plots of the pareto front
            gen_scores = all_gen_score_df.loc[all_gen_score_df['generation'] == i]
            gen_scores = gen_scores.drop(columns=["Unnamed: 0", "generation"])
            # print(gen_scores.min())
            scores = gen_scores.to_numpy(dtype=np.float64)
            all_ind_gen_scores = gen_scores.drop(columns=['2'])
            all_scores = all_ind_gen_scores.to_numpy(dtype=np.float64)
            # print(scores.min())
            # apply function to gather pareto front scores
            pareto = is_pareto(scores)
            pareto_front = scores[pareto]

            pareto_front_df = pd.DataFrame(pareto_front)

            pareto_front_df.sort_values(0, inplace=True)

            x_all = all_scores[:, 0]
            y_all = all_scores[:, 1]
            x_pareto = pareto_front_df.iloc[:, 0]
            y_pareto = pareto_front_df.iloc[:, 1]
            plt.plot(x_pareto, y_pareto, color='r', alpha=0.4, label='pareto front line')
            plt.scatter(x_pareto, y_pareto, marker="x", alpha=0.8, label='non-dominated individuals')
            plt.scatter(x_all, y_all, marker='o', color='orange', alpha=0.2, label='dominated individuals')
            # random pool
            plt.scatter(random_pool_x_pareto, random_pool_y_pareto, marker='+', color='black', alpha=0.3,
                        label='non-dominated random_pool individuals')

            x_limit = (max(all_scores[:, 0]) + 100)
            y_limit = (max(all_scores[:, 1]) + 2500)
            plt.xlim(15000, 23000)
            plt.ylim(4000, 11000)
            title = ax.set_title("\n".join(wrap("Front Plot 3:" + " Repetition " + str(rep) + " Experiment:" +
                                                experiment_name_for_plot +
                                                " Generation " + str(i), 50)))
            plt.tight_layout()
            title.set_y(1.05)
            fig.subplots_adjust(top=0.85)
            plt.xlabel('Metric 1')
            plt.ylabel('Metric 2')
            plt.legend(loc='upper left')
            plt.savefig("output_data/experiment_1/new_2d_all-plot_generation_" + str(i) + " Repetition " + str(rep) +
                        "_pareto_front_3_" + experiment_name,
                        bbox_inches="tight")
            # plt.show()
            # time.sleep(5)
            plt.close()

        for i in range(final_gen):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # create pareto plots of the pareto front
            gen_scores = all_gen_score_df.loc[all_gen_score_df['generation'] == i]
            gen_scores = gen_scores.drop(columns=["Unnamed: 0", "generation"])
            scores = gen_scores.to_numpy()
            all_ind_gen_scores = gen_scores.drop(columns=["1"])
            all_scores = all_ind_gen_scores.to_numpy(dtype=np.float64)
            # apply function to gather pareto front scores
            pareto = is_pareto(scores)
            pareto_front = scores[pareto]

            pareto_front_df = pd.DataFrame(pareto_front)
            pareto_front_df.sort_values(0, inplace=True)

            x_all = all_scores[:, 0]
            y_all = all_scores[:, 1]
            x_pareto = pareto_front_df.iloc[:, 0]
            y_pareto = pareto_front_df.iloc[:, 2]
            plt.plot(x_pareto, y_pareto, color='r', alpha=0.4, label='pareto front line')
            plt.scatter(x_pareto, y_pareto, marker="x", alpha=0.8, label='non-dominated individuals')
            plt.scatter(x_all, y_all, marker='o', color='orange', alpha=0.2, label='dominated individuals')
            # random pool
            plt.scatter(random_pool_x_pareto, random_pool_z_pareto, marker='+', color='black', alpha=0.3,
                        label='non-dominated random_pool individuals')
            x_limit = (max(all_scores[:, 0]) + 100)
            y_limit = (max(all_scores[:, 1]) + 1)
            plt.xlim(10000, 23000)
            plt.ylim(0, 13)
            title = ax.set_title("\n".join(wrap("Front Plot 2:" + " Repetition " + str(rep) + " Experiment:" +
                                                experiment_name_for_plot +
                                                " Generation " + str(i), 50)))
            plt.tight_layout()
            title.set_y(1.05)
            fig.subplots_adjust(top=0.85)
            plt.xlabel('Metric 1')
            plt.ylabel('Metric 3')
            plt.legend(loc='upper left')
            plt.savefig("output_data/experiment_1/new_2d_all-plot_generation_" + str(i) + " Repetition " + str(rep) +
                        "_pareto_front_2_" + experiment_name,
                        bbox_inches="tight")
            # plt.show()
            # time.sleep(5)
            plt.close()

        # as normal; 2 and 3
        for i in range(final_gen):
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # create pareto plots of the pareto front
            gen_scores = all_gen_score_df.loc[all_gen_score_df['generation'] == i]
            gen_scores = gen_scores.drop(columns=["Unnamed: 0", "generation"])
            scores = gen_scores.to_numpy()
            all_ind_gen_scores = gen_scores.drop(columns=["0"])
            all_scores = all_ind_gen_scores.to_numpy(dtype=np.float64)
            # apply function to gather pareto front scores
            pareto = is_pareto(scores)
            pareto_front = scores[pareto]

            pareto_front_df = pd.DataFrame(pareto_front)
            pareto_front_df.sort_values(1, inplace=True)

            x_all = all_scores[:, 0]
            y_all = all_scores[:, 1]
            x_pareto = pareto_front_df.iloc[:, 1]
            y_pareto = pareto_front_df.iloc[:, 2]
            plt.plot(x_pareto, y_pareto, color='r', alpha=0.4, label='pareto front line')
            plt.scatter(x_pareto, y_pareto, marker="x", alpha=0.8, label='non-dominated individuals')
            plt.scatter(x_all, y_all, marker='o', color='orange', alpha=0.2, label='dominated individuals')
            # random pool
            plt.scatter(random_pool_y_pareto, random_pool_z_pareto, marker='+', color='black', alpha=0.3,
                        label='non-dominated random_pool individuals')

            # x_limit = (max(all_scores[:, 0]) + 100)
            # y_limit = (max(all_scores[:, 1]) + 1)
            plt.xlim(4000, 11000)
            plt.ylim(0, 13)
            title = ax.set_title("\n".join(wrap("Front Plot 1:" + " Repetition " + str(rep) + " Experiment:" +
                                                experiment_name_for_plot +
                                                " Generation " + str(i), 50)))
            plt.tight_layout()
            title.set_y(1.05)
            fig.subplots_adjust(top=0.85)
            plt.xlabel('Metric 2')
            plt.ylabel('Metric 3')
            plt.legend(loc='upper left')
            plt.savefig("output_data/experiment_1/new_2d_all-plot_generation_" + str(i) + " Repetition " + str(rep) +
                        "_pareto_front_1_" + experiment_name,
                        bbox_inches="tight")
            # plt.show()
            # time.sleep(3)
            plt.close()
