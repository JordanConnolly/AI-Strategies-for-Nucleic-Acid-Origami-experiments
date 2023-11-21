import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Pandas and Numpy Options
cwd = os.getcwd()

experiment_name = ""
diagnostic = ""
i = 1
inner_loop_rep = i
feature_names = ""

# import df of real and predicted values of folds
real_list = []
pred_list = []

# import shap values
shap = []
shap_values = []
df_test_shap = []

# import actual_vs_predicted
total_actual_vs_pred_df = pd.DataFrame()

# Save folds actual vs predicts -- DIAGNOSTIC --
fig, ax = plt.subplots()
ax.scatter(real_list, pred_list)
ax.loglog([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.title(experiment_name + "_Actual_vs_Predicted_plot_rep_" + str(i + 1) + "_fold_" + str(inner_loop_rep))
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.savefig(diagnostic + experiment_name +
            "_Actual_vs_Predicted_plot_rep_" + str(i + 1) + "_fold_"
            + str(inner_loop_rep) + '.png', bbox_inches="tight")
plt.show()

# Shap summary plot
shap.summary_plot(shap_values, df_test_shap,
                  feature_names=feature_names, show=False)
plt.title(experiment_name + " whole data set Shap summary plot experiment " + str(i + 1)
          + " fold " + str(inner_loop_rep))
plt.savefig(diagnostic + experiment_name +
            "_SHAP_plot_rep_" + str(i + 1) + "_fold_" + str(inner_loop_rep) + '.png', bbox_inches="tight")

# Total Actual vs Predicted RESIDUALS across all folds -- IMPORTANT --
plot_lm_1 = plt.figure()
plot_lm_1.axes[0] = sns.residplot(y=np.log10(total_actual_vs_pred_df['prediction']),
                                  x=np.log10(total_actual_vs_pred_df['reality']), data=total_actual_vs_pred_df,
                                  lowess=True,
                                  scatter_kws={'alpha': 0.5},
                                  line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plt.title(experiment_name + " Residual Plot repetition " + str(i + 1) + " All Folds")
plt.xlabel('Log10: Actual values Magnesium (mM)')
plt.ylabel('Log10: Model Residual values')
plt.savefig(important + experiment_name +
            "_Actual_vs_Predicted_sns_residplot_rep_" + str(i + 1) + "_whole_dataset" + '.png', bbox_inches="tight")
plt.show()

# Total Actual vs Predicted across all folds -- IMPORTANT --
fig, ax = plt.subplots()
ax.scatter(total_actual_vs_pred_df['reality'], total_actual_vs_pred_df['prediction'])
ax.loglog([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.title(experiment_name + " All Folds Actual vs Predicted plot repetition " + str(i + 1))
ax.set_xlabel('Log Actual Magnesium (mM)')
ax.set_ylabel('Log Predicted Magnesium (mM)')
plt.savefig(important + experiment_name +
            "_Total_Actual_vs_Predicted_plot_rep_" + str(i + 1) + '.png', bbox_inches="tight")
plt.show()