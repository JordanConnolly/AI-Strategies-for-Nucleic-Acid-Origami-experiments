import pandas as pd
import numpy as np


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


origami_name = "8__dunn"

rev_pool_pareto_with_index_and_scaffold_score = pd.read_csv("PARETO_SELECTED_SEQUENCES/" + origami_name +
                                                            "_reverse_sets_pareto_front_of_final_sequences_selected"
                                                            "_code.csv")
random_pool_pareto_with_index_and_scaffold_score = pd.read_csv("PARETO_SELECTED_SEQUENCES/" + origami_name +
                                                               "_random_pool_pareto_front_of_final_sequences_selected"
                                                               "_code.csv")
fwd_pool_pareto_with_index_and_scaffold_score = pd.read_csv("PARETO_SELECTED_SEQUENCES/" + origami_name +
                                                            "_forward_sets_pareto_front_of_final_sequences_selected"
                                                            "_code.csv")

random_pool_pareto_with_index_and_scaffold_score_metric = random_pool_pareto_with_index_and_scaffold_score[["Metric 1",
                                                                                                            "Metric 2",
                                                                                                            "Metric 3",
                                                                                                            "Metric 4"]]

fwd_pool_pareto_with_index_and_scaffold_score_metric = fwd_pool_pareto_with_index_and_scaffold_score[["Metric 1",
                                                                                                      "Metric 2",
                                                                                                      "Metric 3",
                                                                                                      "Metric 4"]]

dominated_by_random = 0
for i in range(len(fwd_pool_pareto_with_index_and_scaffold_score_metric)):
    print(i)

    # append the row to rank from forward pool
    row_to_rank = fwd_pool_pareto_with_index_and_scaffold_score_metric.iloc[[i]]
    # random_pool_pareto_with_index_and_scaffold_score_metric_rank = \
    #     random_pool_pareto_with_index_and_scaffold_score_metric.append(row_to_rank)

    # this allows you to check the entire front vs entire front
    random_pool_pareto_with_index_and_scaffold_score_metric_rank = pd.concat([fwd_pool_pareto_with_index_and_scaffold_score_metric,
                                                                              random_pool_pareto_with_index_and_scaffold_score_metric])
    df_for_topsis = random_pool_pareto_with_index_and_scaffold_score_metric_rank.values

    # apply topsis and return result and compare to the row_to_rank
    import topsispy as tp

    # #####################################################
    # Settings for the best alternative selector metrics ##
    # #####################################################
    # calculate the best alternative using TOPSIS implementation
    metrics_set = 4
    reverse_front = False  # will pick the "worst" origami scores

    if metrics_set == 4:
        if reverse_front is True:
            weights = [0.25, 0.25, 0.25, 0.25]
            costs = [1, 1, 1, 1]
        else:
            weights = [0.25, 0.25, 0.25, 0.25]
            costs = [-1, -1, -1, -1]

    if metrics_set == 3:
        if reverse_front is True:
            weights = [0.33, 0.33, 0.33]
            costs = [1, 1, 1]
        else:
            weights = [0.33, 0.33, 0.33]
            costs = [-1, -1, -1]

    # #####################
    # APPLY TOPSIS #######
    # ####################
    # call TOPSIS to make a decision using our pareto_front, the weights and cost parameters
    decision = tp.topsis(df_for_topsis, weights, costs)
    # ind_topsis = pareto_front we calculated[TOPSIS_result[index_position/individual_selected]]
    ind_topsis = df_for_topsis[decision[0]]
    # print("The best individual topsis selected:", ind_topsis)
    # print(row_to_rank)



# append the row to rank from forward pool
row_to_rank = fwd_pool_pareto_with_index_and_scaffold_score_metric.iloc[[i]]
# random_pool_pareto_with_index_and_scaffold_score_metric_rank = \
#     random_pool_pareto_with_index_and_scaffold_score_metric.append(row_to_rank)

# this allows you to check the entire front vs entire front
random_pool_pareto_with_index_and_scaffold_score_metric_rank = pd.concat([fwd_pool_pareto_with_index_and_scaffold_score_metric,
                                                                          random_pool_pareto_with_index_and_scaffold_score_metric])
df_for_topsis = random_pool_pareto_with_index_and_scaffold_score_metric_rank.values

# apply function to gather pareto front scores
pareto = is_pareto(df_for_topsis)
# when a mask is collected (true / false) apply to the scores to retain non-dominated individuals
pareto_front = df_for_topsis[pareto]
# create a data frame to apply a sort of values (for presentation / graph only)
pareto_front_df = pd.DataFrame(pareto_front)
pareto_front = pareto_front_df.values

print("total len of pareto front:", len(pareto_front))

# print(np.intersect1d(pareto_front, random_pool_pareto_with_index_and_scaffold_score_metric, assume_unique=True))
np.array([x for x in set(tuple(x) for x in A) & set(tuple(x) for x in B)])