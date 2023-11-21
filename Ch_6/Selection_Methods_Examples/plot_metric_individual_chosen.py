import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
from textwrap import wrap
import time
import os

# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # SET EXPERIMENT PARAMETERS # # # # # #

experiment_set_name = "CXPB_SWEEP_SET_2/"
# normalisation values for metrics
worst_random_pool = [23124.26209913061, 10334.305573960144, 18.28311920166016]
# create variable for num of individual in gen
individuals = 100
# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # #

# store as csv table
all_hyper_volumes_df = pd.read_csv("../GA_Raw_Results_Analysis/Analysis_Output/" +
                                   experiment_set_name.strip("/") + "_hypervolume_table.csv")
all_gra_df = pd.read_csv("../GA_Raw_Results_Analysis/Analysis_Output/" +
                         experiment_set_name.strip("/") + "_gra_table.csv")
print(all_gra_df.head(5))
print(all_hyper_volumes_df.head(5))
# extract just GRA from the DF
gra_values_to_merge = all_gra_df["best individual metrics via GRA"]
all_hyper_volumes_df = pd.merge(all_hyper_volumes_df, gra_values_to_merge,
                                left_index=True, right_index=True, how='outer')
print(all_hyper_volumes_df.head(5))

# base_path / path to all files in this project
basepath = path.dirname(__file__)

#  create new paths for origami used
storage_path = "Individual_Metric_plots/2d_generational_0_100/experiment_1/" + experiment_set_name
if not os.path.exists(storage_path):
    os.makedirs(storage_path)


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


# access and import the random search scores
rp_path = path.abspath(path.join(basepath, "..", "Pareto_Plots", "random_search_results", "DBS_square",
                                 "DBS_square_RW-experiment_TESTrandom_walk_scaffold_scores.csv"))
random_pool_df = pd.read_csv(rp_path)
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


for experiment in all_hyper_volumes_df.itertuples():
    rep = experiment[3]
    experiment_name = experiment[2]
    print(rep, experiment_name)

    hypervolume_ind_series = experiment[5]
    topsis_ind_series = experiment[6]
    gra_ind_series = experiment[10]

    # print(experiment)
    # print(experiment_name)
    # print(hypervolume_ind_series)
    # print(topsis_ind_series)
    # print(gra_ind_series)

    hypervolume_ind_list = str(hypervolume_ind_series).replace(" ", ",").strip("[").strip("]").split(",")
    while '' in hypervolume_ind_list:
        hypervolume_ind_list.remove('')
    hypervolume_ind = [int(i) if i.isdigit() else float(i) for i in hypervolume_ind_list]

    topsis_ind_list = str(topsis_ind_series).replace(" ", ",").strip("[").strip("]").split(",")
    while '' in topsis_ind_list:
        topsis_ind_list.remove('')
    topsis_ind = [int(i) if i.isdigit() else float(i) for i in topsis_ind_list]

    gra_ind_list = str(gra_ind_series).replace(" ", ",").strip("[").strip("]").split(",")
    while '' in gra_ind_list:
        gra_ind_list.remove('')
    gra_ind = [int(i) if i.isdigit() else float(i) for i in gra_ind_list]

    print(rep)
    print(experiment_name)

    if rep > 0:
        # name of experiment file
        # experiment_name = "CXPB_SWEEP_CXPB_90_MUTATIONS_16BP_INDMUTATION_50"

        # access externally
        filepath = "D:/Generator_RQ4/Systematic_Experiments/" + experiment_set_name + experiment_name

        # access the file of interest for reading
        # basepath = path.dirname(__file__)
        # experiment_name = "CXPB_SWEEP_CXPB_60_MUTATIONS_16BP"
        # filepath = path.abspath(path.join(basepath, "..", "GA_Raw_Results", experiment_name))
        print(filepath)

        input_path = filepath  # find the path for the input data
        metadata_path = input_path + "/metadata/"  # specify the metadata path
        results_path = input_path + "/results/"  # specify the results path
        plot_path = input_path + "/plots/"  # specify the plots path
        origami_name = "DBS_square"

        # set display precision
        pd.set_option("display.precision", 20)
        np.set_printoptions(precision=40, formatter='longfloat')

        # import the paths and name the origami of interest
        # rep = 6
        rep_name = "rep_" + str(rep)

        # you may as well not use the metadata path, and use the compiled results pathways
        # gen0_score_df = pd.read_csv(metadata_path + origami_name + "_GA-experiment_" + rep_name + "_scores_gen_0.csv")
        gen0_scaffold_df = pd.read_csv(metadata_path + origami_name + "_GA-experiment_" +
                                       rep_name + "_scaffolds_gen_0.csv")

        # all gens in loop
        all_gen_score_df = pd.read_csv(results_path + origami_name + "_GA-experiment_" +
                                       rep_name + "_scores_final_gen.csv")  # load all of the generation score
        all_gen_scaffold_df = pd.read_csv(results_path + origami_name + "_GA-experiment_" +
                                          rep_name + "_scaffolds_final_gen.csv")

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
        final_gen_scores = all_gen_score_df.loc[all_gen_score_df['generation'] == final_gen].\
            drop(columns=['Unnamed: 0', 'generation'])

        experiment_name_for_plot = experiment_name.strip("CXPB")
        experiment_name_for_plot = experiment_name_for_plot.strip("_SWEEP_")

        # Front 3
        fig = plt.figure()
        ax = fig.add_subplot(111)
        front_3_gen_scores_0 = all_gen_score_df.loc[all_gen_score_df['generation'] == 0]

        front_3_gen_scores_100 = all_gen_score_df.loc[all_gen_score_df['generation'] == final_gen]

        front_3_gen_scores_0 = front_3_gen_scores_0.drop(columns=["Unnamed: 0", "generation"])
        front_3_gen_scores_100 = front_3_gen_scores_100.drop(columns=["Unnamed: 0", "generation"])
        front_3_gen_scores_0_scores = front_3_gen_scores_0.to_numpy()
        front_3_gen_scores_100_scores = front_3_gen_scores_100.to_numpy()

        # apply function to gather pareto front scores
        front_3_pareto_0 = is_pareto(front_3_gen_scores_0_scores)
        front_3_pareto_front_gen_0 = front_3_gen_scores_0_scores[front_3_pareto_0]

        # apply function to gather pareto front scores
        front_3_pareto_100 = is_pareto(front_3_gen_scores_100_scores)
        front_3_pareto_front_gen_100 = front_3_gen_scores_100_scores[front_3_pareto_100]

        front_3_pareto_front_gen_100_df = pd.DataFrame(front_3_pareto_front_gen_100)
        front_3_pareto_front_gen_100_df.sort_values(0, inplace=True)
        front_3_pareto_front_gen_100_values = front_3_pareto_front_gen_100_df.values

        front_3_pareto_front_gen_0_df = pd.DataFrame(front_3_pareto_front_gen_0)
        front_3_pareto_front_gen_0_df.sort_values(0, inplace=True)
        front_3_pareto_front_gen_0_values = front_3_pareto_front_gen_0_df.values

        front_3_x_all = front_3_pareto_front_gen_0_values[:, 0]
        front_3_y_all = front_3_pareto_front_gen_0_values[:, 1]
        front_3_x_pareto = front_3_pareto_front_gen_100_values[:, 0]
        front_3_y_pareto = front_3_pareto_front_gen_100_values[:, 1]

        # plt.plot(front_3_x_pareto, front_3_y_pareto, color='r', alpha=0.4, label='gen100 pareto-front line')
        plt.scatter(front_3_x_pareto, front_3_y_pareto, marker="x", alpha=0.4, label='non-dominated gen100 individuals')
        plt.scatter(front_3_x_all, front_3_y_all, marker='o', color='orange', alpha=0.8,
                    label='non-dominated gen0 individuals')
        # plt.plot(front_3_x_all, front_3_y_all, color='g', alpha=0.4, label='gen0 pareto-front line')
        # random pool
        plt.scatter(random_pool_x_pareto, random_pool_y_pareto, marker='+', color='black', alpha=0.3,
                    label='non-dominated random_pool individuals')

        # additional metrics
        print(hypervolume_ind[0], hypervolume_ind[1])
        print(topsis_ind[0], topsis_ind[1])
        print(gra_ind[0], gra_ind[1])

        plt.scatter(hypervolume_ind[0], hypervolume_ind[1], marker='v', color='red', alpha=1.0,
                    label='best hypervolume individual')
        plt.scatter(topsis_ind[0], topsis_ind[1], marker='^', color='black', alpha=1.0,
                    label='best TOPSIS individual')
        plt.scatter(gra_ind[0], gra_ind[1], marker='<', color='purple', alpha=1.0,
                    label='best GRA individual')

        # plt.plot(random_pool_x_pareto, random_pool_y_pareto, color='black', alpha=0.4, label='gen0 pareto-front line')
        ###
        x_limit = (max(all_gen_score_df["0"]) + 100)
        y_limit = (max(all_gen_score_df["1"]) + 2500)
        plt.xlim(15000, x_limit)
        plt.ylim(4000, y_limit)
        title = ax.set_title("\n".join(wrap("Front Plot 3:" + " Repetition " + str(rep) + " Experiment:" +
                                            experiment_name_for_plot +
                                            " Generation 0 vs 100", 50)))
        title.set_y(1.05)
        fig.subplots_adjust(top=0.85)
        plt.tight_layout()
        plt.xlabel('Metric 1')
        plt.ylabel('Metric 2')
        plt.legend(loc='upper left', framealpha=0.4)
        plt.savefig(storage_path + "2d_all-plot_Generation_0_vs_100_pareto_front_3_"
                    + " " + rep_name + experiment_name, bbox_inches="tight")
        # plt.show()
        # time.sleep(5)
        plt.close()

        # Front 2
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # create pareto plots of the pareto front
        front_2_gen_scores_0 = all_gen_score_df.loc[all_gen_score_df['generation'] == 0]
        front_2_gen_scores_100 = all_gen_score_df.loc[all_gen_score_df['generation'] == final_gen]
        front_2_gen_scores_0 = front_2_gen_scores_0.drop(columns=["Unnamed: 0", "generation"])
        front_2_gen_scores_100 = front_2_gen_scores_100.drop(columns=["Unnamed: 0", "generation"])
        front_2_gen_scores_0_scores = front_2_gen_scores_0.to_numpy()
        front_2_gen_scores_100_scores = front_2_gen_scores_100.to_numpy()

        # apply function to gather pareto front scores
        front_2_pareto_0 = is_pareto(front_2_gen_scores_0_scores)
        front_2_pareto_front_gen_0 = front_2_gen_scores_0_scores[front_2_pareto_0]

        # apply function to gather pareto front scores
        front_2_pareto_100 = is_pareto(front_2_gen_scores_100_scores)
        front_2_pareto_front_gen_100 = front_2_gen_scores_100_scores[front_2_pareto_100]

        front_2_pareto_front_gen_100_df = pd.DataFrame(front_2_pareto_front_gen_100)
        front_2_pareto_front_gen_100_df.sort_values(0, inplace=True)
        front_2_pareto_front_gen_100_values = front_2_pareto_front_gen_100_df.values

        front_2_pareto_front_gen_0_df = pd.DataFrame(front_2_pareto_front_gen_0)
        front_2_pareto_front_gen_0_df.sort_values(0, inplace=True)
        front_2_pareto_front_gen_0_values = front_2_pareto_front_gen_0_df.values

        x_all = front_2_pareto_front_gen_0_values[:, 0]
        y_all = front_2_pareto_front_gen_0_values[:, 2]
        x_pareto = front_2_pareto_front_gen_100_values[:, 0]
        y_pareto = front_2_pareto_front_gen_100_values[:, 2]

        # plt.plot(x_pareto, y_pareto, color='r', alpha=0.4, label='gen100 pareto-front line')
        plt.scatter(x_pareto, y_pareto, marker="x", alpha=0.4, label='non-dominated gen100 individuals')
        plt.scatter(x_all, y_all, marker='o', color='orange', alpha=0.8, label='non-dominated gen0 individuals')
        # plt.plot(x_all, y_all, color='g', alpha=0.4, label='gen0 pareto-front line')
        # random pool
        plt.scatter(random_pool_x_pareto, random_pool_z_pareto, marker='+', color='black', alpha=0.3,
                    label='non-dominated random_pool individuals')

        # additional metrics
        print(hypervolume_ind[0], hypervolume_ind[2])
        print(topsis_ind[0], topsis_ind[2])
        print(gra_ind[0], gra_ind[2])

        plt.scatter(hypervolume_ind[0], hypervolume_ind[2], marker='v', color='red', alpha=1.0,
                    label='best hypervolume individual')
        plt.scatter(topsis_ind[0], topsis_ind[2], marker='^', color='black', alpha=1.0,
                    label='best TOPSIS individual')
        plt.scatter(gra_ind[0], gra_ind[2], marker='<', color='purple', alpha=1.0,
                    label='best GRA individual')

        x_limit = (max(all_gen_score_df["0"]) + 100)
        y_limit = (max(all_gen_score_df["2"]) + 1)
        plt.xlim(10000, x_limit)
        plt.ylim(0, y_limit)
        title = ax.set_title("\n".join(wrap("Front Plot 2:" + " Repetition " + str(rep) + " Experiment:" +
                                            experiment_name_for_plot +
                                            " Generation 0 vs 100", 50)))
        title.set_y(1.05)
        fig.subplots_adjust(top=0.85)
        plt.xlabel('Metric 1')
        plt.ylabel('Metric 3')
        plt.legend(loc='upper left', framealpha=0.4)
        plt.tight_layout()
        plt.savefig(storage_path + "2d_all-plot_Generation_0_vs_100_pareto_front_2_"
                    + " " + rep_name + experiment_name, bbox_inches="tight")
        # plt.show()
        # time.sleep(5)
        plt.close()

        # Front 1
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # create pareto plots of the pareto front
        front_1_gen_scores_0 = all_gen_score_df.loc[all_gen_score_df['generation'] == 0]
        front_1_gen_scores_100 = all_gen_score_df.loc[all_gen_score_df['generation'] == final_gen]
        front_1_gen_scores_0 = front_1_gen_scores_0.drop(columns=["Unnamed: 0", "generation"])
        front_1_gen_scores_100 = front_1_gen_scores_100.drop(columns=["Unnamed: 0", "generation"])
        front_1_gen_scores_0_scores = front_1_gen_scores_0.to_numpy()
        front_1_gen_scores_100_scores = front_1_gen_scores_100.to_numpy()

        # apply function to gather pareto front scores
        front_1_pareto_0 = is_pareto(front_1_gen_scores_0_scores)
        front_1_pareto_front_gen_0 = front_1_gen_scores_0_scores[front_1_pareto_0]

        # apply function to gather pareto front scores
        front_1_pareto_100 = is_pareto(front_1_gen_scores_100_scores)
        front_1_pareto_front_gen_100 = front_1_gen_scores_100_scores[front_1_pareto_100]

        front_1_pareto_front_gen_100_df = pd.DataFrame(front_1_pareto_front_gen_100)
        front_1_pareto_front_gen_100_df.sort_values(0, inplace=True)
        front_1_pareto_front_gen_100_values = front_1_pareto_front_gen_100_df.values

        front_1_pareto_front_gen_0_df = pd.DataFrame(front_1_pareto_front_gen_0)
        front_1_pareto_front_gen_0_df.sort_values(0, inplace=True)
        front_1_pareto_front_gen_0_values = front_1_pareto_front_gen_0_df.values

        front_1_x_all = front_1_pareto_front_gen_0_values[:, 1]
        front_1_y_all = front_1_pareto_front_gen_0_values[:, 2]
        front_1_x_pareto = front_1_pareto_front_gen_100_values[:, 1]
        front_1_y_pareto = front_1_pareto_front_gen_100_values[:, 2]

        # plt.plot(front_1_x_pareto, front_1_y_pareto, color='r', alpha=0.4, label='gen100 pareto-front line')
        plt.scatter(front_1_x_pareto, front_1_y_pareto, marker="x", alpha=0.4, label='non-dominated gen100 individuals')
        plt.scatter(front_1_x_all, front_1_y_all, marker='o', color='orange', alpha=0.8, label='non-dominated gen0 individuals')
        # plt.plot(front_1_x_all, front_1_y_all, color='g', alpha=0.4, label='gen0 pareto-front line')
        # random pool
        plt.scatter(random_pool_y_pareto, random_pool_z_pareto, marker='+', color='black', alpha=0.3,
                    label='non-dominated random_pool individuals')

        # additional metrics
        print(hypervolume_ind[1], hypervolume_ind[2])
        print(topsis_ind[1], topsis_ind[2])
        print(gra_ind[1], gra_ind[2])

        plt.scatter(hypervolume_ind[1], hypervolume_ind[2], marker='v', color='red', alpha=1.0,
                    label='best hypervolume individual')
        plt.scatter(topsis_ind[1], topsis_ind[2], marker='^', color='black', alpha=1.0,
                    label='best TOPSIS individual')
        plt.scatter(gra_ind[1], gra_ind[2], marker='<', color='purple', alpha=1.0,
                    label='best GRA individual')

        x_limit = (max(all_gen_score_df["1"]) + 100)
        y_limit = (max(all_gen_score_df["2"]) + 1)
        plt.xlim(4000, x_limit)
        plt.ylim(0, y_limit)
        title = ax.set_title("\n".join(wrap("Front Plot 1:" + " Repetition " + str(rep) + " Experiment:" +
                                            experiment_name_for_plot +
                                            " Generation 0 vs 100", 50)))
        title.set_y(1.05)
        fig.subplots_adjust(top=0.85)
        plt.xlabel('Metric 2')
        plt.ylabel('Metric 3')
        plt.legend(loc='upper left', framealpha=0.4)
        plt.tight_layout()
        plt.savefig(storage_path + "2d_all-plot_Generation_0_vs_100_pareto_front_1_"
                    + experiment_name + " " + rep_name, bbox_inches="tight")
        # plt.show()
        # time.sleep(3)
        plt.close()
