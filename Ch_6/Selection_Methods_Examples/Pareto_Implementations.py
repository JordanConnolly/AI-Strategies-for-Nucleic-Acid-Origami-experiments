import numpy as np


def is_pareto(costs, maximise=False):
    # source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    :param costs: An (n_points, n_costs) array
    :maximise: boolean. True for maximising, False for minimising
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(costs[is_efficient] >= c, axis=1)  # Remove dominated points
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)  # Remove dominated points
    return is_efficient


# def is_pareto(costs):
#     # source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
#     """
#     Find the pareto-efficient points
#     :param costs: An (n_points, n_costs) array
#     :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
#     """
#     is_efficient = np.ones(costs.shape[0], dtype=bool)
#     for i, c in enumerate(costs):
#         if is_efficient[i]:
#             is_efficient[is_efficient] = np.any(costs[is_efficient] <= c, axis=1)  # Keep any point with a lower cost
#             is_efficient[i] = True  # And keep self
#     return is_efficient


# def is_pareto(scores):
#     # Source: https://pythonhealthcare.org/tag/pareto-front/
#     # Count number of items
#     population_size = scores.shape[0]
#     # Create a NumPy index for scores on the pareto front (zero indexed)
#     population_ids = np.arange(population_size)
#     # Create a starting list of items on the Pareto front
#     # All items start off as being labelled as on the Pareto front
#     pareto_front = np.ones(population_size, dtype=bool)
#     # Loop through each item. This will then be compared with all other items
#     for i in range(population_size):
#         # Loop through all other items
#         for j in range(population_size):
#             # Check if our 'i' point is dominated by out 'j' point
#             if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
#                 # j dominates i. Label 'i' point as not on Pareto front
#                 pareto_front[i] = 0
#                 # Stop further comparisons with 'i' (no more comparisons needed)
#                 break
#     # Return ids of scenarios on pareto front
#     return population_ids[pareto_front]
# # Fairly fast for many datapoints, less fast for many costs, somewhat readable
