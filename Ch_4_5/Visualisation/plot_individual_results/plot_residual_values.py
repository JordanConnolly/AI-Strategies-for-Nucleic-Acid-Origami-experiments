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

original_data_frame = pd.read_excel("")
features = original_data_frame.columns

# import actual_vs_predicted
total_actual_vs_pred_df = pd.DataFrame()

# Total Actual vs Predicted RESIDUALS across all folds -- IMPORTANT --
plot_lm_1 = plt.figure()
plot_lm_1.axes[0] = sns.residplot(y=np.log10(total_actual_vs_pred_df['prediction']),
                                  x=np.log10(total_actual_vs_pred_df['reality']), data=total_actual_vs_pred_df,
                                  lowess=True,
                                  scatter_kws={'alpha': 0.5},
                                  line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plt.title(experiment_name + " Residual Plot repetition " + repetition + " All Folds")
plt.xlabel('Log10: Actual values Magnesium (mM)')
plt.ylabel('Log10: Model Residual values')
plt.savefig(cwd + "_Actual_vs_Predicted_sns_residplot_rep_" + str(inner_loop_rep)
            + "_whole_dataset" + '.png', bbox_inches="tight")
plt.show()
