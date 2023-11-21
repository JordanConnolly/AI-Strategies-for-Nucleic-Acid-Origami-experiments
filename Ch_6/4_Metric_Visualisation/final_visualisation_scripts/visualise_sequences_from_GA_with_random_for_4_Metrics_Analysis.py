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


import re


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


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


# base_path / path to all files in this project
basepath = path.dirname(__file__)

# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # SET EXPERIMENT PARAMETERS # # # # #
origami_name_list = ["8__dunn"]
all_origami_experiment_list = ["CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_50"]


# find the median and best hypervolume repetitions
for i in range(len(origami_name_list)):
    # set the origami for plotting
    origami_name = origami_name_list[i]
    experiment_list = [all_origami_experiment_list[i]]
    experiment_set_name = origami_name + "_SWEEP/"
    # create variable for num of individual in gen
    individuals = 40
    median_value = 4
    # calculate the best alternative using TOPSIS implementation
    weights = [0.25, 0.25, 0.25, 0.25]
    costs = [-1, -1, -1, -1]
    # access and import the random search scores
    random_scores_path = "D:/Generator_RQ4/Systematic_Experiments/" + origami_name + \
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

    # apply function to gather pareto front scores
    random_pool_dropped = random_pool_df.drop(columns=["Unnamed: 0"])
    random_pool_scores = random_pool_dropped.to_numpy(dtype=np.float64)
    random_pool_pareto = is_pareto(random_pool_scores)
    random_pool_pareto_front = random_pool_scores[random_pool_pareto]
    random_pool_pareto_front_df = pd.DataFrame(random_pool_pareto_front)
    random_pool_pareto_front = random_pool_pareto_front_df.values

    for experiment_name in experiment_list:
        # access externally
        filepath = "D:/Generator_RQ4/Systematic_Experiments/" + experiment_set_name + experiment_name
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
                scores = gen_scores.to_numpy(dtype=np.float64)
                # apply function to gather pareto front scores
                pareto = is_pareto(scores)
                pareto_front = scores[pareto]
                pareto_front_df = pd.DataFrame(pareto_front)
                pareto_front_df.sort_values(0, inplace=True)
                pareto_front = pareto_front_df.values
                hv_calculated = wti_hv.calc(scores)
                hyper_volumes.append([experiment_name, rep, hv_calculated])

        # create a df to analyse and get best/median, before adding best and median to total experiment df
        hyper_volume_df = pd.DataFrame(hyper_volumes)
        hyper_volume_df.columns = ["experiment", "rep", "hypervolume"]
        hyper_volume_df.sort_values(by=['hypervolume'], inplace=True, ascending=False)
        # best and med reps
        best_rep_df = hyper_volume_df.iloc[0, :].copy()
        med_rep_df = hyper_volume_df.iloc[median_value, :].copy()

        best_repetition = best_rep_df["rep"]
        median_repetition = med_rep_df["rep"]

        # create a data frame of the scaffolds from median repetition
        all_gen_scaffold_df = pd.DataFrame()
        for scaffold_gen in range(1, 251, 1):
            # get the scaffold of a generation
            scaffold_df = pd.read_csv(metadata_path + origami_name + "_GA-experiment_rep_" +
                                      str(median_repetition) + "_scaffolds_gen_" + str(scaffold_gen) + ".csv")
            # concatenate it into a data frame
            scaffold_df["generation"] = scaffold_gen
            all_gen_scaffold_df = pd.concat([all_gen_scaffold_df, scaffold_df])

        # calculate sequence length and base positions
        single_nucleotide_sequence = all_gen_scaffold_df.iloc[0, 1]
        sequence_base_pos_list = []
        # add base position
        for base_pos_number in range(1, len(single_nucleotide_sequence)+1):
            sequence_base_pos_list.append(base_pos_number)

        # encoded the sequences
        from sklearn.preprocessing import LabelEncoder


        def ordinal_encoder(data_to_encode):
            my_array = np.array(list(data_to_encode))
            label_encoder = LabelEncoder()
            label_encoder.fit(np.array(['A', 'C', 'G', 'T', 'U']))
            integer_encoded = label_encoder.transform(my_array)
            return integer_encoded


        encoded_seq_list = []
        encoded_sequence = ordinal_encoder(single_nucleotide_sequence)
        for encoded_number in encoded_sequence:
            encoded_seq_list.append(encoded_seq_list)

        plt.plot(encoded_seq_list, sequence_base_pos_list)
        plt.show()
