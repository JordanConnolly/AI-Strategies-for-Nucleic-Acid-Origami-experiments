from pymoo.factory import get_performance_indicator
import pandas as pd
import numpy as np
import pandas as pd
from os import path
from Pareto_Implementations import is_pareto
import topsispy as tp


# #####################################################
# Settings for the best alternative selector metrics ##
# #####################################################
# calculate the best alternative using TOPSIS implementation
metrics_set = 4
reverse_front = True  # will pick the "worst" origami scores

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

# #######################################################
# Import final generation data, calculate pareto front ##
# #######################################################
# import final generation data (could be read using built-in reader, I use pandas)
gen_scores = pd.read_csv("example_final_generation_scores_four_metric.csv")
gen_scores.drop(columns=["Unnamed: 0"], inplace=True)
scores = gen_scores.to_numpy(dtype=np.float64)

# apply function to gather pareto front scores
pareto = is_pareto(scores)
# when a mask is collected (true / false) apply to the scores to retain non-dominated individuals
pareto_front = scores[pareto]
# create a data frame to apply a sort of values (for presentation / graph only)
pareto_front_df = pd.DataFrame(pareto_front)
pareto_front_df.sort_values(0, inplace=True)
pareto_front = pareto_front_df.values

# #####################
# APPLY TOPSIS #######
# ####################
# call TOPSIS to make a decision using our pareto_front, the weights and cost parameters
decision = tp.topsis(pareto_front, weights, costs)
# ind_topsis = pareto_front we calculated[TOPSIS_result[index_position/individual_selected]]
ind_topsis = pareto_front[decision[0]]
print("The best individual topsis selected:", ind_topsis)


# print(decision)  # returns array where [0] returns the position in list of best option

######################
