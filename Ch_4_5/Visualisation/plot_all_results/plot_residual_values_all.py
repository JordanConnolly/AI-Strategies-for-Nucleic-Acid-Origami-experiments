import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re

# Directories containing files
cwd = os.getcwd()
path = "C:/Users/Big JC/PycharmProjects/PhDCodeOnly/ML_results_to_analyse/" \
       "investigating_instance_rem_dna/extra_trees_results/model_plots/"
experiment_name = "Extra_Trees_RFE_3CV_Remove_Buffer_and_DNA"
numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


residual_files = sorted(glob.glob(path + experiment_name + "_actual_vs_pred_rep_" + "*" + "_best_all_folds.csv"
                                  ), key=numerical_sort)

for file_number in range(len(residual_files)):
    file = residual_files[file_number+1]
    path, filename = os.path.split(file)
    print(filename)
    rep_regex = r"rep_\d*"
    rep_found = re.findall(rep_regex, file)[0]
    repetition = rep_found.split("_")[1]
    # fold_regex = r"fold_\d*"
    # fold_found = re.findall(fold_regex, file)[0]
    # inner_loop_rep = fold_found.split("_")[1]

    total_actual_vs_pred_df = pd.read_csv(file)
    total_actual_vs_pred_df = total_actual_vs_pred_df.drop(columns="Unnamed: 0")
    print(total_actual_vs_pred_df)
    prediction_values = total_actual_vs_pred_df['prediction'].values
    actual_values = total_actual_vs_pred_df['reality'].values

    # Total Actual vs Predicted RESIDUALS across all folds -- IMPORTANT --
    plot_lm_1 = plt.figure()
    plot_lm_1.axes[0] = sns.residplot(y=np.log10(total_actual_vs_pred_df['prediction']),
                                      x=np.log10(total_actual_vs_pred_df['reality']), data=total_actual_vs_pred_df,
                                      # lowess=True,
                                      scatter_kws={'alpha': 0.5})
                                      # line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    # plt.title(experiment_name + " Residual Plot repetition " + str(repetition))
    # plt.xlabel('Log10: Actual values Magnesium (mM)')
    # plt.ylabel('Log10: Model Residual values')
    # plt.xlim(5, 32)
    # plt.ylim(10, -10)
    # plt.savefig(cwd + "_Actual_vs_Predicted_sns_residplot_rep_" + str(inner_loop_rep)
    #             + "_whole_dataset" + '.png', bbox_inches="tight")
    plt.show()
    plt.pause(1)
