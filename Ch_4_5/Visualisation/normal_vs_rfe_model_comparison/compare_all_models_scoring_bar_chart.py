import pandas as pd
import glob
import os
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

path = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/" \
       "magnesium_experiments/machine-learning/results_path/exhaustive_round_robin_results/" \
       "normal_vs_rfe_model_comparison/all_results"  # use your path
all_files = glob.glob(path + "/*_.csv")

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
frame.sort_values(by=['r2'], inplace=True, ascending=False)
frame.to_csv('report_comparison_of_models.csv')
print(frame)

storage_path = "C:/Users/Big JC/PycharmProjects/MachineLearningPhD/" \
               "magnesium_experiments/machine-learning/results_path/" \
               "exhaustive_round_robin_results/normal_vs_rfe_model_comparison/compared_results_stored/"

r2_frame = frame[frame.r2 > -0.5]
plt.barh(r2_frame['filename'], r2_frame['r2'], alpha=0.5, align='center')
plt.xlim([-0.5, 1])
plt.title('r2 values')
plt.tight_layout()
plt.savefig(storage_path + "model_comparison_r2.png", figsize=(20, 10))
plt.show()

plt.barh(r2_frame['filename'], r2_frame['mae'], alpha=0.5, align='center')
plt.title('mae values')
plt.tight_layout()
plt.savefig(storage_path + "model_comparison_mae.png", figsize=(20, 10))
plt.show()

plt.barh(r2_frame['filename'], r2_frame['mse'], alpha=0.5, align='center')
plt.title('mse values')
plt.tight_layout()
plt.savefig(storage_path + "model_comparison_mse.png", figsize=(20, 10))
plt.show()

plt.barh(r2_frame['filename'], r2_frame['rmse'], alpha=0.5, align='center')
plt.title('rmse values')
plt.tight_layout()
plt.savefig(storage_path + "model_comparison_rmse.png", figsize=(20, 10))
plt.show()

plt.barh(r2_frame['filename'], r2_frame['medianae'], alpha=0.5, align='center')
plt.title('medianae values')
plt.tight_layout()
plt.savefig(storage_path + "model_comparison_medianae.png", figsize=(20, 10))
plt.show()
