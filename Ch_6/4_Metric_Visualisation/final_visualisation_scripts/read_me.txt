Create spreadsheets of results from EXPERIMENTAL RESULTS files:
1) run file: random_pool_pareto_front_analysis_remove_wti_normalisation.py
2) run file: experiments_imported_hypervolume_analysis_remove_wti_normalisation.py
3) run file: present_results_of_hypervolume_analysis_remove_wti_normalisation.py

this creates the spreadsheet tables with results for each origami
then do the following to produce plots:

4) create PCP plot of Pareto Front: PCP_Plots_with_random_for_4_Metrics_Analysis.py
5) create PCP plot of individuals: PCP_Plots_with_random_for_4_Metrics_Analysis_selection.py
6) create Pairwise Scatter plot of Pareto Front: PW_Scatter_Plots_with_random_for_4_Metrics_Analysis.py

If you have REVERSE PROBLEM EXPERIMENTAL RESULTS files;
then you can also produce the reverse pareto front files:
7) run steps 2 and 3 but this time using the reverse experiments
8) create reverse PCP plot of Pareto Front: reverse_comparison_PCP_Plots_with_random_for_4_Metrics_Analysis.py
9) create reverse PCP plots of individuals: reverse_comparison_PCP_Plots_with_random_for_4_Metrics_Analysis_selection.py

You should now have the tables and plots for Forwards and Reverse problem origamis with a random 100,000 selector pool.
