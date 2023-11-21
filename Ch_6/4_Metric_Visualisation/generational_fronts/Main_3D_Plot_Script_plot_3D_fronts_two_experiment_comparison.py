from os import path
from textwrap import wrap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

experiment_set_name = "CXPB_SWEEP_SET_12"
individuals = 40

# base_path / path to all files in this project
basepath = path.dirname(__file__)

# access and import the random search scores
rp_path = path.abspath(path.join(basepath, "..", "Pareto_Plots", "random_search_results", "DBS_square",
                                 "DBS_square_all_300000_evaluations.csv"))
random_pool_df = pd.read_csv(rp_path)
print(random_pool_df)
drop_list = ["0.0", "1.0", "2.0"]
random_pool_df = random_pool_df[~random_pool_df.isin(drop_list)]

# access and import the details for the plot from a table
filepath = path.abspath(path.join(basepath, "..", "GA_Raw_Results_Analysis", "Analysis_Output",
                                  (experiment_set_name + "_non_norm_total_table.csv")))
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

# experiment_list = [26]
# for experiment in experiment_list:
for experiment in table_df.itertuples():
    if experiment[2] != "CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_25":
        print("Skip")
    else:
        rep = experiment[3]
        experiment_name = experiment[2]
        if rep > 0:
            # # name of experiment file
            # experiment_name = "CXPB_SWEEP_CXPB_30_MUTATIONS_16BP_INDMUTATION_25"
            experiment_name_for_plot = experiment_name.strip("CXPB")
            experiment_name_for_plot = experiment_name_for_plot.strip("_SWEEP_")
            # access externally
            filepath = "D:/Generator_RQ4/Systematic_Experiments/CXPB_3_Metrics/DBS_square/" + experiment_set_name + "/" + \
                       experiment_name
            print(experiment_name)
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
            # rep = experiment
            rep_name = "rep_" + str(rep)

            # you may as well not use the metadata path, and use the compiled results pathways
            # gen0_score_df = pd.read_csv(metadata_path + origami_name + "_GA-experiment_" + rep_name + "_scores_gen_0.csv")
            gen0_scaffold_df = pd.read_csv(
                metadata_path + origami_name + "_GA-experiment_" + rep_name + "_scaffolds_gen_0.csv")

            # all gens in loop
            all_gen_score_df = pd.read_csv(
                results_path + origami_name + "_GA-experiment_" + rep_name + "_scores_final_gen.csv")
            # load all of the generation score
            all_gen_scaffold_df = pd.read_csv(
                results_path + origami_name + "_GA-experiment_" + rep_name + "_scaffolds_final_gen.csv")

            # create variable for num of individual in gen
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
            final_gen_scores = all_gen_score_df.loc[all_gen_score_df['generation'] == final_gen].drop(
                columns=['generation'])

            df_selected_generations = pd.DataFrame()
            for i in range(0, final_gen + 1, 5):
                # create pareto plots of the pareto front
                gen_scores = all_gen_score_df.loc[all_gen_score_df['generation'] == i]
                gen_scores_dropped = gen_scores.drop(columns=["Unnamed: 0", "generation"])
                # calculate pareto scores
                scores = gen_scores_dropped.to_numpy(dtype=np.float64)
                pareto = is_pareto(scores)
                pareto_front = scores[pareto]
                pareto_front_df = pd.DataFrame(pareto_front)
                pareto_front_df['generation'] = i
                df_selected_generations = pd.concat([pareto_front_df, df_selected_generations])

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            x_pareto = df_selected_generations.iloc[:, 0]
            y_pareto = df_selected_generations.iloc[:, 1]
            z_pareto = df_selected_generations.iloc[:, 2]

            gen = df_selected_generations.iloc[:, 3]

            p = ax.scatter(x_pareto, y_pareto, z_pareto, s=20, c=gen, marker='x', cmap='Spectral', alpha=0.3,
                           label='non-dominated individuals')
            # ax.scatter(x_all, y_all, z_all, marker='x', s=20, c="blue", alpha=0.1, label='dominated individuals')
            # random pool
            ax.scatter(random_pool_x_pareto, random_pool_y_pareto, random_pool_z_pareto,
                       s=20, c="black", alpha=1, label='random pool non-dominated individuals')

            plt.title("3D Pareto Front:" + rep_name + experiment_name)
            # title = ax.set_title("\n".join(wrap("3D Pareto Front:" + " Repetition " + str(rep) + " Experiment:" +
            #                                     experiment_name_for_plot, 50)))
            title = ax.set_title("\n".join(wrap("Generational Pareto Front Plot of Median Hypervolume Experiment from "
                                                "Parameter Sweep 12", 50)))
            plt.xlabel('Metric 1')
            plt.ylabel('Metric 2')
            ax.set_zlabel('Metric 4')
            ax.legend(loc=(0, 3 / 4))
            ax.yaxis.labelpad = 30
            fig.colorbar(p, shrink=0.8, pad=0.10, label='Generation')
            plt.tight_layout()

            # # automatically set upper limits
            # x_limit = (max(all_gen_score_df["0"]) + 100)
            # y_limit = (max(all_gen_score_df["1"]) + 1000)
            # z_limit = (max(all_gen_score_df["2"]))

            # manually set upper limits
            x_limit = 22000
            y_limit = 12000
            z_limit = 10
            plt.xlim(16000, x_limit)
            plt.ylim(6000, y_limit)
            ax.set_zlim(0, z_limit)

            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            ax.zaxis.set_tick_params(labelsize=8)
            # plt.show()

            plt.savefig("output_data_generational_plot/experiment_1/3d_front-plot_generation_" +
                        "Experiment_" + experiment_name_for_plot + "repetition_" + str(rep) + ".png",
                        bbox_inches="tight")

            import time
            time.sleep(1)

            # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            # ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            # plt.xticks(rotation=45)
