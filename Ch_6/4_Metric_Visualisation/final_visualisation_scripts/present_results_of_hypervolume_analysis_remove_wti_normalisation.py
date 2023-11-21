from os import path
import pandas as pd
import ast
import numpy as np

# open both tables, create one single data frame, split into two tables, one for inds; one for whole experiment
first_table_columns = ["experiment", "rep", "hypervolume", "repetition_type"]

list_of_origami_names = ['ball', '6hb', 'dunn', 'DBS_square',
                         'minitri', 'Abrick', 'fourfinger-circular',
                         'fourfinger-linear',
                         'nanoribbonRNA', 'hj']

for origami_name in list_of_origami_names:
    experiment_set_names = [origami_name + "_Sweep"]


    # base_path / path to all files in this project
    basepath = path.dirname(__file__)
    # access and import
    stored_path = path.abspath(path.join(basepath, "Analysis_Output/"))


    for i in range(len(experiment_set_names)):
        experiment_set = experiment_set_names[i]
        # initial tables import
        total_df = pd.read_csv(stored_path + "/" + experiment_set + "_non_norm_total_table.csv")
        # result tables
        total_result = pd.DataFrame(total_df[first_table_columns])

        # create pivoted tables, where experiments are concatenated, showing best and median rep values
        total_result['experiment'] = total_result['experiment'].str.strip("CXPB_SWEEP_")
        total_result['experiment'] = 'CXPB_' + total_result['experiment'].astype(str)

        total_table = total_result.pivot(index='experiment', columns=['repetition_type'], values=['hypervolume', 'rep'])

        total_table.columns = range(total_table.shape[1])
        total_table.columns = ["best hypervolume", "median hypervolume", "best repetition", "median repetition"]

        # store two tables, one for hyper-volume, one for individual details (pareto front individual selected)
        total_table.to_csv("Analysis_Output/" + experiment_set.strip("/") + "_non_norm_hypervolume_presentation_table.csv")
        # ind_table.to_csv("Analysis_Output/" + experiment_set.strip("/") + "_non_norm_individual_table.csv")
