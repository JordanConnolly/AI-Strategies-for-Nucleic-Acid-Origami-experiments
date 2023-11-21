import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import animation
import time
from os import path
import re
import glob
from textwrap import wrap

# access and import the details for the plot from a table
basepath = path.dirname(__file__)
# access and import the details for the plot from a table
filepath = path.abspath(path.join(basepath, "..", "GA_Raw_Results_Analysis", "Analysis_Output",
                                  "CXPB_SWEEP_SET_12_non_norm_total_table.csv"))
table_df = pd.read_csv(filepath)

# for experiment in experiment_list:
for experiment in table_df.itertuples():
    rep = experiment[3]
    experiment_name = experiment[2]
    print(experiment_name)
    if rep > 0:
        # # name of experiment file
        # experiment_name = "CXPB_SWEEP_CXPB_90_MUTATIONS_16BP_INDMUTATION_50"

        experiment_name_for_plot = experiment_name.strip("CXPB")
        experiment_name_for_plot = experiment_name_for_plot.strip("_SWEEP_")

        # access externally
        # filepath = "D:/Generator_RQ4/Systematic_Experiments/CXPB_3_Metrics/CXPB_SWEEP_SET_2/" + experiment_name

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
        # rep = 17
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


        df_selected_generations = pd.DataFrame()
        for i in range(0, final_gen+1, 5):
            # create pareto plots of the pareto front
            print(i)
            gen_scores = all_gen_score_df.loc[all_gen_score_df['generation'] == i]
            gen_scores_dropped = gen_scores.drop(columns=["Unnamed: 0", "generation"])
            # calculate pareto scores
            scores = gen_scores_dropped.to_numpy(dtype=np.float64)
            pareto = is_pareto(scores)
            pareto_front = scores[pareto]
            pareto_front_df = pd.DataFrame(pareto_front)
            pareto_front_df['generation'] = i
            df_selected_generations = pd.concat([pareto_front_df, df_selected_generations])

        print(df_selected_generations)
        print(df_selected_generations.columns)

        from mpl_toolkits import mplot3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_pareto = df_selected_generations.iloc[:, 0]
        y_pareto = df_selected_generations.iloc[:, 1]
        z_pareto = df_selected_generations.iloc[:, 2]

        gen = df_selected_generations.iloc[:, 3]

        p = ax.scatter(x_pareto, y_pareto, z_pareto, s=20, c=gen, cmap='Spectral',
                       alpha=1, label='non-dominated individuals')
        # ax.scatter(x_all, y_all, z_all, marker='x', s=20, c="blue", alpha=0.1, label='dominated individuals')
        # plt.title("3D Pareto Front:" + rep_name + experiment_name)
        # title = ax.set_title("\n".join(wrap("3D Pareto Front:" + " Repetition " + str(rep) + " Experiment:" +
        #                                     experiment_name_for_plot, 50)))
        title = ax.set_title("\n".join(wrap("3D Plot: Best Experiment from Parameter Sweep 2; "
                                            "Pareto Front Plot showing every 5 generations", 50)))
        plt.xlabel('Metric 1')
        plt.ylabel('Metric 2')
        ax.set_zlabel('Metric 3')
        ax.legend(loc=(0, 3/4))
        fig.colorbar(p, shrink=0.8, pad=0.10)
        plt.tight_layout()

        x_limit = (max(all_gen_score_df["0"]) + 100)
        y_limit = (max(all_gen_score_df["1"]) + 1000)
        z_limit = (max(all_gen_score_df["2"]))
        plt.xlim(16000, x_limit)
        plt.ylim(6000, y_limit)
        ax.set_zlim(0, z_limit)
        plt.savefig("output_data_generational_plot/experiment_1/3d_front-plot_generation" +
                    "_repetition_" + str(rep) + "_Experiment_" + experiment_name_for_plot, bbox_inches="tight")
        # plt.show()
        # time.sleep(5)

        # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        # ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        # plt.xticks(rotation=45)
