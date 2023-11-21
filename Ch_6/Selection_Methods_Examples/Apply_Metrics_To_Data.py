from pymoo.factory import get_performance_indicator
import pandas as pd
import numpy as np
import pandas as pd
from os import path
from GRA_metric import GrayRelationalCoefficient, plot_average_grey_relational_coefficient
from Pareto_Implementations import is_pareto
import topsispy as tp


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


# normalisation values for metrics
worst_random_pool = [23124.26209913061, 10334.305573960144, 18.28311920166016]

# #####################################################
# Settings for the best alternative selector metrics ##
# #####################################################

# create performance indicator for use later
hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2, 1.2]))

# calculate the best alternative using TOPSIS implementation
weights = [0.33, 0.33, 0.33]
costs = [-1, -1, -1]

# #####################################################


# #######################################################
# Import final generation data, calculate pareto front ##
# #######################################################

# import final generation data (could be read using built-in reader, I use pandas)
gen_scores = pd.read_csv("example_final_generation_scores.csv")
gen_scores.drop(columns=["Unnamed: 0"], inplace=True)
# normalise results to between 0 and 1 using worst random pool values possible
# this is useful for the Hyper-parameter presentation (values are smaller)
gen_scores["0"] = gen_scores["0"] / worst_random_pool[0]
gen_scores["1"] = gen_scores["1"] / worst_random_pool[1]
gen_scores["2"] = gen_scores["2"] / worst_random_pool[2]
# print(gen_scores)

scores = gen_scores.to_numpy(dtype=np.float64)
# apply function to gather pareto front scores
pareto = is_pareto(scores)
# when a mask is collected (true / false) apply to the scores to retain non-dominated individuals
pareto_front = scores[pareto]
# create a data frame to apply a sort of values (for presentation / graph only)
pareto_front_df = pd.DataFrame(pareto_front)
pareto_front_df.sort_values(0, inplace=True)
pareto_front = pareto_front_df.values

hv_calculated = hv.calc(scores)
print("This is the hyper-volume for the entire final generation:", hv_calculated)

# #####################################################


# ######################################################################
# Iterate over individuals in pareto set: calculate selection metrics ##
# ######################################################################

######################
# iterates over each individual on the pareto front, calculate hyper-volumes (hyper-lines, really)
non_dom_len = len(pareto_front)
ind_list = []
hv_for_ind = []
for inds_i in range(non_dom_len):
    check_best_ind = pareto_front[inds_i]
    check_best_ind_hv = hv.calc(check_best_ind)
    # append to lists
    hv_for_ind.append(check_best_ind_hv)
    ind_list.append(check_best_ind)

# you could implement this without pandas also
hyper_volume_ind_df = pd.DataFrame(hv_for_ind, columns=['individuals_hypervolume_values'])
hyper_volume_ind_df["individual metrics"] = ind_list
hyper_volume_ind_df.reset_index(inplace=True)
hyper_volume_ind_df = hyper_volume_ind_df.sort_values(by='individuals_hypervolume_values', ascending=False)
print("The best individual hyper-volume selected hv:", hyper_volume_ind_df["individuals_hypervolume_values"].iloc[0])
print("The best individual hyper-volume selected:", un_normalise_metrics(hyper_volume_ind_df["individual metrics"].iloc[0], worst_random_pool))
######################

######################
# APPLY TOPSIS
decision = tp.topsis(pareto_front, weights, costs)
ind_topsis = pareto_front[decision[0]]
# print("The best individual topsis selected:", ind_topsis)
print("The best individual topsis selected:", un_normalise_metrics(ind_topsis, worst_random_pool))
######################

######################
# APPLY GRA
# Establish a gray correlation model, standardize data
best_rep_gra_model = GrayRelationalCoefficient(pareto_front, standard=False)
best_rep_cors = best_rep_gra_model.get_calculate_relational_coefficient()

# create data frame to sort and index the values obtained
best_rep_gra_df = pd.DataFrame()
best_rep_max_corr = max(best_rep_cors, key=lambda x: x.tolist())
# retrieve index of best
best_rep_max_index_ndarray = np.argwhere(best_rep_cors == best_rep_max_corr)
best_rep_max_index_list_ndarray = best_rep_max_index_ndarray.tolist()
best_rep_max_index_list = [item[0] for item in best_rep_max_index_list_ndarray]
best_rep_max_index_duplicate = set([x for x in best_rep_max_index_list if best_rep_max_index_list.count(x) > 1])
best_rep_max_index = best_rep_max_index_duplicate.pop()
best_rep_best_gra_ind = pareto_front[best_rep_max_index]
best_rep_gra_norm = un_normalise_metrics(best_rep_best_gra_ind, worst_random_pool)
print("The best individual GRA selected:", best_rep_gra_norm)
######################
