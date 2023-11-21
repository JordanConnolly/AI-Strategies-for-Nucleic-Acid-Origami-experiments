from os import path
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# base_path / path to all files in this project
basepath = path.dirname(__file__)

# access and import the random search scores
rp_path = path.abspath(path.join(basepath, "..", "Pareto_Plots", "random_search_results", "DBS_square",
                                 "DBS_square_all_300000_evaluations.csv"))
random_pool_df = pd.read_csv(rp_path)
print(random_pool_df)
drop_list = ["0.0", "1.0", "2.0"]
random_pool_df = random_pool_df[~random_pool_df.isin(drop_list)]


# def is_pareto(costs, maximise=False):
#     # source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
#     """
#     :param costs: An (n_points, n_costs) array
#     :maximise: boolean. True for maximising, False for minimising
#     :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
#     """
#     is_efficient = np.ones(costs.shape[0], dtype=bool)
#     for i, c in enumerate(costs):
#         if is_efficient[i]:
#             if maximise:
#                 is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)  # Remove dominated points
#             else:
#                 is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)  # Remove dominated points
#     return is_efficient


def is_pareto(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


# worst_random_pool = [23124.26209913061, 10334.305573960144, 18.28311920166016]
worst_random_pool = [23725.61823468173, 10470.082988426608, 21.17147827148437]  # 300,000 seq random pool
random_pool_dropped = random_pool_df.drop(columns=["Unnamed: 0"])

# # normalise results to between 0 and 1 using worst random pool values possible
# random_pool_dropped["0"] = random_pool_dropped["0"] / worst_random_pool[0]
# random_pool_dropped["1"] = random_pool_dropped["1"] / worst_random_pool[1]
# random_pool_dropped["2"] = random_pool_dropped["2"] / worst_random_pool[2]


# calculate pareto scores
rp_scores = random_pool_dropped.to_numpy(dtype=np.float64)
rp_pareto = is_pareto(rp_scores)
rp_pareto_front = rp_scores[rp_pareto]
rp_pareto_front_df = pd.DataFrame(rp_pareto_front, dtype=np.float64)
print(rp_pareto_front_df)

random_pool_scores = random_pool_dropped.to_numpy(dtype=np.float64)
random_pool_pareto = is_pareto(random_pool_scores)
random_pool_pareto_front = random_pool_scores[random_pool_pareto]

random_pool_pareto_front = pd.DataFrame(random_pool_pareto_front)
random_pool_pareto_front.sort_values(0, inplace=True)
random_pool_pareto_front = random_pool_pareto_front.values

random_pool_x_pareto = random_pool_pareto_front[:, 0]
random_pool_y_pareto = random_pool_pareto_front[:, 1]
random_pool_z_pareto = random_pool_pareto_front[:, 2]

random_pool_x = random_pool_scores[:, 0]
random_pool_y = random_pool_scores[:, 1]
random_pool_z = random_pool_scores[:, 2]

# # 2D PLOTS!!!
# print(len(random_pool_x_pareto))
# print(len(random_pool_y_pareto))
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# # plt.plot(random_pool_x_pareto, random_pool_y_pareto, color='r', alpha=0.4, label='pareto front line')
# plt.scatter(random_pool_x_pareto, random_pool_y_pareto, marker="x", alpha=1.0, label='non-dominated individuals ("optimal" scaffolds)')
# plt.scatter(random_pool_x, random_pool_y, marker='+', color='orange', alpha=0.4, label='dominated individuals (worse than optimal scaffolds)')
# plt.xlabel("Metric j")
# plt.ylabel("Metric i")
# title = ax.set_title("\n".join(wrap("Example 2D Plot of Two Metrics for 10,000 Scaffolds with Calculated Pareto Front", 50)))
# title.set_y(1.05)
# fig.subplots_adjust(top=0.85)
# plt.tight_layout()
# plt.legend(loc='upper left', framealpha=1.0)
# plt.show()
#


# # 3D PLOTS!!!
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# random pool
ax.scatter(random_pool_x_pareto, random_pool_y_pareto, random_pool_z_pareto,
           s=20, c="blue", marker='x', alpha=1, label='dominated individuals')
ax.scatter(random_pool_x, random_pool_y, random_pool_z,
           s=20, c="black", alpha=0.08, marker='o', label='random pool non-dominated individuals')

plt.title("3D Pareto Front: DBS_square")
title = ax.set_title("\n".join(wrap("3D Pareto Front: 300,000 Random Scaffolds DBS_square", 50)))
plt.xlabel('Metric 1')
plt.ylabel('Metric 2')
ax.set_zlabel('Metric 3')
ax.legend(loc=(0, 3 / 4))
plt.tight_layout()

# # automatically set upper limits
# x_limit = (max(all_gen_score_df["0"]) + 100)
# y_limit = (max(all_gen_score_df["1"]) + 1000)
# z_limit = (max(all_gen_score_df["2"]))

# manually set upper limits
x_limit = 24000
y_limit = 15000
z_limit = 22
plt.xlim(16000, x_limit)
plt.ylim(6000, y_limit)
ax.set_zlim(0, z_limit)
plt.show()

#
# from matplotlib import animation
# # random pool gif
#
#
# def init():
#     # random pool
#     pareto_scatter = ax.scatter(random_pool_x_pareto, random_pool_y_pareto, random_pool_z_pareto,
#                                 s=20, c="blue", marker='x', alpha=1, label='random pool non-dominated individuals')
#     ax.scatter(random_pool_x_pareto, random_pool_y_pareto, random_pool_z_pareto,
#                s=20, c="blue", marker='x', alpha=1, label='random pool non-dominated individuals')
#     all_scatter = ax.scatter(random_pool_x, random_pool_y, random_pool_z,
#                              s=20, c="black", alpha=0.08, marker='o', label='random pool dominated individuals')
#     ax.scatter(random_pool_x, random_pool_y, random_pool_z,
#                s=20, c="black", alpha=0.08, marker='o', label='random pool dominated individuals')
#
#     plt.title("3D Pareto Front: DBS_square")
#     title = ax.set_title("\n".join(wrap("3D Pareto Front: 10,000 Random Scaffolds DBS_square", 50)))
#     plt.xlabel('Metric 1')
#     plt.ylabel('Metric 2')
#     ax.set_zlabel('Metric 3')
#     ax.legend(loc=(0, 3 / 4))
#     plt.tight_layout()
#     plt.title(title)
#     ax.legend(loc=(0, 3 / 4), handles=[pareto_scatter, all_scatter])
#     # plt.tight_layout()
#     return fig,
#
#
# def animate(i):
#     ax.view_init(elev=20., azim=i)
#     return fig,
#
#
# # Animate
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=360, interval=10000, blit=True)
# # Save
# anim.save('gif_output_data/all_pareto_3d_animation_Random_Pool_DBS_square' + '.gif', fps=30)
