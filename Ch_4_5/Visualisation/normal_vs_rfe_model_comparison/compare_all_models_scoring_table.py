import pandas as pd
import glob
import os
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})


rr_path = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/" \
       "magnesium_experiments/machine-learning/results_path/exhaustive_round_robin_results/" \
       "normal_vs_rfe_model_comparison/all_results"  # use your path

normal_path = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/" \
              "magnesium_experiments/machine-learning/results_path" \
              "/04052020_results/normal_vs_rfe_model_comparison/all_results/"

storage_path = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/make_life_easy/store_comparison_table"


def create_data_frame(path_to_file, sort_by):
    all_files = glob.glob(path_to_file + "/*.csv")

    li = []
    names = ['r2', 'mae', 'mse', 'rmse', 'medianae']

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, names=names)
        file_base_name = os.path.basename(filename)
        model_name_split = file_base_name.split(".")
        model_name = model_name_split[0]
        df['filename'] = model_name
        df = df.iloc[[0]]
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.sort_values(by=[sort_by], inplace=True, ascending=False)
    return frame


round_robin_table = (create_data_frame(rr_path, 'filename'))
print(create_data_frame(rr_path, 'filename'))
round_robin_table.to_csv("round_robin_results.csv")
normal_experiment_table = (create_data_frame(normal_path, 'filename'))
print(create_data_frame(normal_path, 'filename'))
normal_experiment_table.to_csv("normal_results.csv")
