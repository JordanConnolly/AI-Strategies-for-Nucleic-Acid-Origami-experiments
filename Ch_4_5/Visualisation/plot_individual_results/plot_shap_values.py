import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# import shap values
shap = []

# Directories containing files
cwd = os.getcwd()
diagnostic = "C:/example"
experiment_name = "Extra Trees RFE"
repetition = str(1)
inner_loop_rep = 1

original_data_frame = pd.read_excel("data")
feature_names = original_data_frame.columns.tolist()
y = ["Magnesium (mM)"]

# Shap summary plot
shap.summary_plot(shap_values, df_test_shap,
                  feature_names=x_column_numpy, show=False)
plt.title(experiment_name + " whole data set Shap summary plot experiment " + str(i + 1)
          + " fold " + str(inner_loop_rep))
plt.savefig(diagnostic + experiment_name +
            "_SHAP_plot_rep_" + str(i + 1) + "_fold_" + str(inner_loop_rep) + '.png', bbox_inches="tight")