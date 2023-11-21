import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap

input_path = "./input_data/CPU_Clock_Check/Test_2/time/"
# gen_0_time_df = pd.read_csv(input_path + "DBS_square_GA-experiment_rep_1_gen_time.txt", header=None)
# gen_1_time_df = pd.read_csv(input_path + "DBS_square_GA-experiment_rep_1_rep_1_gen_time.txt", header=None)
# total_time_df = pd.read_csv(input_path + "DBS_square_GA-experiment_rep_1_total_run_time.txt", header=None)

# gen_0_time_df = pd.read_csv(input_path + "DBS_square_GA-experiment_rocket_gen_0.txt", header=None)
# gen_1_time_df = pd.read_csv(input_path + "DBS_square_GA-experiment_rocket_gen_1.txt", header=None)
# total_time_df = pd.read_csv(input_path + "DBS_square_GA-experiment_rocket_total_run_time.txt", header=None)

# gen_0_time_df = pd.read_csv(input_path + "DBS_square_GA-experiment_rocket_gen_0_20ind.txt", header=None)
# gen_1_time_df = pd.read_csv(input_path + "DBS_square_GA-experiment_rocket_gen_1_20ind.txt", header=None)
total_time_df = pd.read_csv(input_path + "DBS_square_GA-experiment_rocket_total_run_time_20ind.txt", header=None)

# time_list = gen_0_time_df
# time_list = gen_1_time_df
time_list = total_time_df
y = []
x = []
counter_x = 0
for i, j in time_list.itertuples():
    counter_x += 2
    print(j)
    new_j = j.split(":")[1]
    y_value = int(float(new_j))
    y.append(y_value)
    x.append(counter_x)


fig, ax = plt.subplots(1)

# plot the data
ax.plot(x, y)
# gen_0_title = "Rocket CPU Cores used vs Run Time: Gen 0 (initialise), 20 individuals"
# gen_1_title = "Rocket CPU Cores used vs Run Time: Gen 1 (1 loop), 20 individuals"
total_time_title = "Rocket CPU Cores used vs Run Time: Total Time (Gen0,Gen1), 20 individuals"

# gen_0_title = "Personal CPU Cores used vs Run Time: Gen 0 (initialise), 20 individuals"
# gen_1_title = "Personal CPU Cores used vs Run Time: Gen 1 (1 loop), 20 individuals"
# total_time_title = "Personal CPU Cores used vs Run Time: Total Time (Gen0,Gen1), 20 individuals"

title = ax.set_title("\n".join(textwrap.wrap(text=total_time_title, width=42)))
plt.tight_layout()
title.set_y(1.05)
fig.subplots_adjust(top=0.8)
plt.ylabel('Run Time (s)')
plt.xlabel('CPU Cores Used')
ax.set_yticks(np.arange(0, max(y)+100, 50))
ax.set_xticks(np.arange(0, max(x)+1, 2))
plt.show()
