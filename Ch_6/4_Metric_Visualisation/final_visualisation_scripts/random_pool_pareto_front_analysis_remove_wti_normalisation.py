import pandas as pd
import numpy as np
from os import path
import glob
from pymoo.factory import get_performance_indicator
from GRA_metric import GrayRelationalCoefficient, plot_average_grey_relational_coefficient
import topsispy as tp  # implementation that is used, doesn't matter which is used
import re


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# base_path / path to all files in this project
basepath = path.dirname(__file__)

list_of_origami_names = ['ball', '6hb', 'dunn', 'DBS_square',
                         'minitri', 'Abrick', 'fourfinger-circular',
                         'fourfinger-linear',
                         'nanoribbonRNA', 'hj']

origami_name = "hj"

# access and import the random search scores
random_scores_path = "E:/PhD Files/RQ4/4_Metrics_SWEEP_Final_Energy_Model/" + origami_name + \
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
random_pool_df.fillna(0, inplace=True)
new_worst_random_pool = []
new_worst_random_pool.append(max(random_pool_df.iloc[:, 1].values))
new_worst_random_pool.append(max(random_pool_df.iloc[:, 2].values))
new_worst_random_pool.append(max(random_pool_df.iloc[:, 3].values))
new_worst_random_pool.append(max(random_pool_df.iloc[:, 4].values))
print(new_worst_random_pool)
worst_random_pool_df = pd.DataFrame(new_worst_random_pool)
worst_random_pool_df.to_csv("worst_theoretical_individual/" +
                            origami_name + "_Worst_Theoretical_Individual.csv")

# normalisation values for metrics
worst_theoretical_individual = new_worst_random_pool  # 100,000 seq random pool
print(worst_theoretical_individual)

# create performance indicator for use later
hv = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2, 1.2, 1.2]))
wti_hv = get_performance_indicator("hv", ref_point=np.array([worst_theoretical_individual[0],
                                                             worst_theoretical_individual[1],
                                                             worst_theoretical_individual[2],
                                                             worst_theoretical_individual[3]]))
# calculate the best alternative using TOPSIS implementation
weights = [0.25, 0.25, 0.25, 0.25]
costs = [-1, -1, -1, -1]


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


# normalise results to between 0 and 1 using worst random pool values,
# use worst theoretical individual normalised data-frame and find the pareto front for ind selection
random_pool_df_wti_norm = random_pool_df.copy()
random_pool_df_wti_norm["0"] = random_pool_df["0"].values / worst_theoretical_individual[0]
random_pool_df_wti_norm["1"] = random_pool_df["1"].values / worst_theoretical_individual[1]
random_pool_df_wti_norm["2"] = random_pool_df["2"].values / worst_theoretical_individual[2]
random_pool_df_wti_norm["3"] = random_pool_df["3"].values / worst_theoretical_individual[3]

random_pool_dropped_wti_norm = random_pool_df_wti_norm.drop(columns=["Unnamed: 0"])
random_pool_scores_wti_norm = random_pool_dropped_wti_norm.to_numpy(dtype=np.float64)
random_pool_pareto_wti_norm = is_pareto(random_pool_scores_wti_norm)
random_pool_pareto_front_wti_norm = random_pool_scores_wti_norm[random_pool_pareto_wti_norm]
headers = ["0", "1", "2", "3"]
random_pool_pareto_front_df_wti_norm = pd.DataFrame(random_pool_pareto_front_wti_norm, columns=[headers])
random_pool_scores_wti_norm = random_pool_pareto_front_df_wti_norm.to_numpy(dtype=np.float64)

# calculate hyper-volume
print("random_pool hyper-volume:", hv.calc(random_pool_scores_wti_norm))
random_pool_calculated = hv.calc(random_pool_scores_wti_norm)


# EXTRA
# Un-normalise metrics to pick a good individual
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


# apply function to gather pareto front scores
random_pool_dropped = random_pool_df.drop(columns=["Unnamed: 0"])
random_pool_scores = random_pool_dropped.to_numpy(dtype=np.float64)
random_pool_pareto = is_pareto(random_pool_scores)
random_pool_pareto_front = random_pool_scores[random_pool_pareto]
random_pool_pareto_front_df = pd.DataFrame(random_pool_pareto_front)
random_pool_pareto_front_df.sort_values(0, inplace=True)
random_pool_pareto_front = random_pool_pareto_front_df.values
random_pool_decision = tp.topsis(random_pool_pareto_front, weights, costs)
print(random_pool_decision)
random_pool_topsis_list = random_pool_pareto_front[random_pool_decision[0]]

# look at individuals on the front
random_pool_non_dom_len = len(random_pool_pareto_front)
random_pool_inds_hv_list = []
for j_random_pool_inds in range(random_pool_non_dom_len):
    check_random_pool_ind = random_pool_pareto_front[j_random_pool_inds]
    check_random_pool_ind_hv = wti_hv.calc(check_random_pool_ind)
    random_pool_inds_hv_list.append(check_random_pool_ind_hv)

# get a sorted index list of the hyper-volume experiments, use idx 0 to get best
check_random_pool_sort_idx = sorted(range(len(random_pool_inds_hv_list)),
                                    key=random_pool_inds_hv_list.__getitem__, reverse=True)
# apply the idx 0 to get the best individual on the pareto front
best_random_pool_ind_hv = random_pool_inds_hv_list[check_random_pool_sort_idx[0]]
best_random_pool_ind = random_pool_pareto_front[check_random_pool_sort_idx[0]]
# for presentation purposes: un-normalised metrics
# best_random_pool_ind_metrics = un_normalise_metrics(best_random_pool_ind, worst_theoretical_individual)
print("Hyper-Line IND:", best_random_pool_ind)

# apply for the random_pool
random_pool_gra_model = GrayRelationalCoefficient(random_pool_pareto_front, standard=False)
random_pool_cors = random_pool_gra_model.get_calculate_relational_coefficient()
random_pool_max_corr = max(random_pool_cors, key=lambda x: x.tolist())
# retrieve index of best
random_pool_max_index_ndarray = np.argwhere(random_pool_cors == random_pool_max_corr)
random_pool_max_index_list_ndarray = random_pool_max_index_ndarray.tolist()
random_pool_max_index_list = [item[0] for item in random_pool_max_index_list_ndarray]
random_pool_max_index_duplicate = set([x for x in random_pool_max_index_list if random_pool_max_index_list.count(x) > 1])
random_pool_max_index = random_pool_max_index_duplicate.pop()
random_pool_best_gra_ind = random_pool_pareto_front[random_pool_max_index]
# random_pool_gra_norm = un_normalise_metrics(random_pool_best_gra_ind, worst_theoretical_individual)
print("GRA IND:", random_pool_best_gra_ind)


def create_metric_list(individual):
    individual_metrics = []
    counter = 0
    for i in range(len(individual)):
        if counter == i:
            add_to_list = individual[i]
            individual_metrics.append(add_to_list)
        counter += 1
    return individual_metrics


hv_individual_to_store = create_metric_list(best_random_pool_ind)
gra_individual_to_store = create_metric_list(random_pool_best_gra_ind)
topsis_individual_to_store = create_metric_list(random_pool_topsis_list)

random_pool_values = [origami_name + "random_pool", 0, random_pool_calculated, hv_individual_to_store,
                      topsis_individual_to_store, best_random_pool_ind_hv, len(check_random_pool_sort_idx),
                      "random_pool"]
random_pool_gra_values = [origami_name + "random_pool", 0, random_pool_calculated,
                          random_pool_max_corr, gra_individual_to_store, "random_pool"]

# store random pool values as separate experiment
hv_columns = ['experiment', 'rep', 'hypervolume',
              'best individual metrics via hypervolume',
              'best individual metrics via topsis', 'best individual hypervolume',
              'pareto front length', 'repetition_type']
random_pool_hyper_volume_df = pd.DataFrame(random_pool_values)
random_pool_hyper_volume_df = random_pool_hyper_volume_df.transpose()
random_pool_hyper_volume_df.columns = hv_columns

gra_columns = ['experiment', 'rep', 'hypervolume', 'best individual GRA Correlations',
               'best individual metrics via GRA', 'repetition_type']
random_pool_gra_df = pd.DataFrame(random_pool_gra_values)
random_pool_gra_df = random_pool_gra_df.transpose()
random_pool_gra_df.columns = gra_columns

random_pool_total_df = pd.merge(random_pool_hyper_volume_df, random_pool_gra_df, how='inner')

# open both tables, create one single data frame, split into two tables, one for inds; one for whole experiment
first_table_columns = ["experiment", "rep", "hypervolume", "repetition_type"]

second_table_columns = ["experiment", "best individual metrics via hypervolume", "best individual metrics via topsis",
                        "best individual hypervolume", "best individual metrics via GRA", "repetition_type"]

# result tables
total_df = random_pool_total_df
total_result = pd.DataFrame(total_df[first_table_columns])
ind_result = pd.DataFrame(total_df[second_table_columns])

# create pivoted tables, where experiments are concatenated, showing best and median rep values
total_result['experiment'] = total_result['experiment'].str.strip("CXPB_SWEEP_")
total_result['experiment'] = 'CXPB_' + total_result['experiment'].astype(str)
ind_result['experiment'] = ind_result['experiment'].str.strip("CXPB_SWEEP_")
ind_result['experiment'] = 'CXPB_' + ind_result['experiment'].astype(str)
total_table = total_result.pivot(index='experiment', columns=['repetition_type'], values=['hypervolume', 'rep'])
ind_table = ind_result.pivot(index='experiment', columns=['repetition_type'], values=[
    "best individual metrics via hypervolume", "best individual metrics via topsis",
    "best individual hypervolume", "best individual metrics via GRA"])

# store two tables, one for hyper-volume, one for individual details (pareto front individual selected)
# total_table.to_csv("Analysis_Output/Random_Walk_300000_remove_norm_topsis_gra_hypervolume_table.csv")
# ind_table.to_csv("Analysis_Output/Random_Walk_300000_remove_norm_topsis_gra_individual_table.csv")
random_pool_total_df.to_csv("Analysis_Output/" + origami_name + "_Selector_remove_norm_topsis_gra_total_table.csv")
