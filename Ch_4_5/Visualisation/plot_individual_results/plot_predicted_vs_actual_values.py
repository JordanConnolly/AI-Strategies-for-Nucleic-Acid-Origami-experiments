import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Directories containing files
cwd = os.getcwd()
diagnostic = "C:/example"
experiment_name = "Extra Trees RFE"
repetition = str(1)
inner_loop_rep = 1

original_data_frame = pd.read_excel("data")
y = ["Magnesium (mM)"]

# import df of real and predicted values of folds
real_list = []
pred_list = []

# Save folds actual vs predicts -- DIAGNOSTIC --
fig, ax = plt.subplots()
ax.scatter(real_list, pred_list)
ax.loglog([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.title(experiment_name + "_Actual_vs_Predicted_plot_rep_" + repetition + "_fold_" + str(inner_loop_rep))
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.savefig(diagnostic + experiment_name +
            "_Actual_vs_Predicted_plot_rep_" + repetition + "_fold_"
            + str(inner_loop_rep) + '.png', bbox_inches="tight")
plt.show()
