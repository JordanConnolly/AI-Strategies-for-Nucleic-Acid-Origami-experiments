import pygraphviz as pgv
from pygraphviz import *
import networkx as nx
from networkx import *
import os
import glob
import re
import pandas as pd
import csv

# Directories containing files
cwd = os.getcwd()
path = cwd + ""

numbers = re.compile(r'(\d+)')


def numerical_sort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def global_file_iteration(path_to_file):
    all_files = sorted(glob.glob(path_to_file + "/*.dot"), key=numerical_sort)
    return all_files


"""
Assuming all origami's are produced in a similar representation to the 
test file: rothemund_rect_triangle
We should be able to calculate around 17 features
"""

revnano_dotfile_path = "E:/PhD Files/RQ3/All_RevNano_Related_Work/outputs/1903_revnano_30mins_midfiles/_mid/"
dot_files = global_file_iteration(revnano_dotfile_path)
names = ['nodes', 'edges', 'avg_neighbour_total', 'graph_density', 'graph_transitivity',
         'average_shortest_path', 'average_clustering_coefficient',
         'average_degree', 'average_betweenness_centrality', 'average_closeness_centrality',
         'graph_assortivity', 'graph_diameter', 'graph_reciprocity', 's-metric', 'wiener_index']
store_features_list = []


for file in dot_files:
    features_list = []
    file_name = file.split("/")
    print(file_name)
    print(file)
    # Import using pygraphviz
    G = pgv.AGraph(file)

    # Swap over to networkx so it doesn't break
    g = nx.Graph(G)

    # general info of origami graph
    features_list.append(nx.number_of_nodes(g))
    features_list.append(nx.number_of_edges(g))

    avg_neighbour_deg = nx.average_neighbor_degree(g)
    avg_neighbour_total = sum(list(avg_neighbour_deg.values())) / len(avg_neighbour_deg)
    features_list.append(avg_neighbour_total)

    # density of the graph
    features_list.append(nx.density(g))

    # transitivity of graph
    features_list.append(nx.transitivity(g))

    # Centrality of Staples origami (density is the average of this)
    degree_centrality_list = [nx.algorithms.degree_centrality(g)]

    # Average Connectivity of Staples origami
    degree_connectivity_list = [nx.average_degree_connectivity(g)]

    # Average shortest path length of origami
    features_list.append(nx.average_shortest_path_length(g))

    origami_ud = g.to_undirected()
    # clustering coefficient of origami graph
    clust_coef = nx.clustering(origami_ud)
    # average clustering coefficient of origami graph
    avg_clust = sum(clust_coef.values()) / len(clust_coef)
    # print("average_clust: ", avg_clust)
    features_list.append(avg_clust)

    # average degrees of origami graph
    N, K = g.order(), g.size()
    avg_deg = float(K) / N
    # print("average_degree: ", avg_deg)
    features_list.append(avg_deg)

    # betweenness centrality of graph edges
    bet_cen = nx.betweenness_centrality(g)
    # print("betweenness_centrality : ", bet_cen)
    bet_cen_values = sum(list(bet_cen.values())) / len(bet_cen)
    # print("average bet cen: ", bet_cen_values)
    features_list.append(bet_cen_values)

    # closeness centrality of graph
    clo_cen = nx.closeness_centrality(g)
    # print("closeness_centrality : ", clo_cen)
    clo_cen_values = sum(list(clo_cen.values())) / len(clo_cen)
    # print("average clo cen: ", clo_cen_values)
    features_list.append(clo_cen_values)

    # top nodes (staples) calculated from centrality measures
    # sort_bet_cen = sorted(bet_cen, reverse=True)
    # top_bet_cen = sort_bet_cen[:50]
    #
    # sort_clo_cen = sorted(bet_cen, reverse=True)
    # top_clo_cen = sort_clo_cen[:50]
    #
    # print(top_bet_cen)
    # print(top_clo_cen)

    # calculate the assortivity value of the graph
    # print("assortivity value: ", nx.algorithms.degree_assortativity_coefficient(g))  # interesting
    features_list.append(nx.algorithms.degree_assortativity_coefficient(g))

    # calculate graph diameter
    # print("diameter of graph: ", nx.algorithms.distance_measures.diameter(g))
    features_list.append(nx.algorithms.distance_measures.diameter(g))

    # calculate graph reciprocity
    # print("graph reciprocity: ", nx.algorithms.overall_reciprocity(g))
    features_list.append(nx.algorithms.overall_reciprocity(g))

    # calculate the rich club coefficient (not very good in protein interactions, maybe not good here
    # print("rich-club coef: ", nx.algorithms.rich_club_coefficient(g))

    """
    A small world network is characterized by a small average shortest path length, and a large clustering coefficient.
    Small-worldness is commonly measured with
    the coefficient sigma or omega.
    Both coefficients compare the average clustering coefficient and shortest path length of a given graph
    against the same quantities for an
    equivalent random or lattice graph.

    NOTE: These seem quite computationally expensive to calculate
    """
    # print("small-world-ness sigma: ", nx.algorithms.sigma(g))
    # print("small-world-ness omega: ", nx.algorithms.omega(g))

    '''
    https://www.researchgate.net/publication/224366095_Measuring_the_
    effectiveness_of_the_s-metric_to_produce_better_network_models

    In order to better distinguish between these networks, the metric s was introduced to measure
    how interconnected the hub nodes are in a network.
    '''
    # print("s-metric: ", nx.algorithms.s_metric(g, normalized=''))
    features_list.append(nx.algorithms.s_metric(g, normalized=''))

    # https://en.wikipedia.org/wiki/Wiener_index
    # print("Weiner index: ", nx.algorithms.wiener_index(g))
    features_list.append(nx.algorithms.wiener_index(g))

    # linalg calculations
    '''https://networkx.github.io/documentation/stable/reference/linalg.html#module-networkx.linalg.modularitymatrix
    Algebraic connectivity and fiedler vectors of undirected graphs
    adjacency eigenvalues
    laplacian eigenvalues
    bethe hessian eigenvalues
    normalized laplace eigenvalues
    modularity spectrum eigenvalues
    '''

    # print("alg connectivity: ", nx.linalg.algebraicconnectivity.algebraic_connectivity(g))
    # print("fiedler vector: ", nx.linalg.algebraicconnectivity.fiedler_vector(g))
    # print("spectral ordering: ", nx.linalg.algebraicconnectivity.spectral_ordering(g))
    # print("adjacency spectrum: ", nx.linalg.spectrum.adjacency_spectrum(g))
    # print("laplacian spectrum: ", nx.linalg.spectrum.laplacian_spectrum(g))
    # print("bethe hessian spectrum: ", nx.linalg.spectrum.bethe_hessian_spectrum(g))
    # print("norm laplace spectrum: ", nx.linalg.spectrum.normalized_laplacian_spectrum(g))
    # print("modularity spectrum: ", nx.linalg.spectrum.modularity_spectrum(g))

    # append features to total list
    store_features_list.append(features_list)

# # store all features of all experiments into a total list
dotfile_df = pd.DataFrame(store_features_list, columns=[names])
dotfile_df.to_csv("1903_all_dot_file_features.csv")
print(dotfile_df.columns)
print(dotfile_df.shape)
print(dotfile_df.head(5))


# for file in dot_files:
#     dotfile_df = pd.DataFrame()
#     features_list = []
#     file_name = file.split("\\")
#     print(file_name)
#
#     # Import using pygraphviz
#     G = pgv.AGraph(file)
#
#     # Swap over to networkx so it doesn't break
#     g = nx.Graph(G)
#
#     # general info of origami graph
#     dotfile_df['nodes'] = (nx.number_of_nodes(g))
#     dotfile_df['edges'] = (nx.number_of_edges(g))
#
#     avg_neighbour_deg = nx.average_neighbor_degree(g)
#     avg_neighbour_total = sum(list(avg_neighbour_deg.values())) / len(avg_neighbour_deg)
#     dotfile_df['avg_neighbour_total'] = avg_neighbour_total
#
#     # density of the graph
#     dotfile_df['graph_density'] = (nx.density(g))
#
#     # transitivity of graph
#     dotfile_df['graph_transitivity'] = (nx.transitivity(g))
#
#     # Centrality of Staples origami (density is the average of this)
#     degree_centrality_list = [nx.algorithms.degree_centrality(g)]
#
#     # Average Connectivity of Staples origami
#     degree_connectivity_list = [nx.average_degree_connectivity(g)]
#
#     # Average shortest path length of origami
#     dotfile_df['average_shortest_path'] = (nx.average_shortest_path_length(g))
#
#     origami_ud = g.to_undirected()
#     # clustering coefficient of origami graph
#     clust_coef = nx.clustering(origami_ud)
#     # average clustering coefficient of origami graph
#     avg_clust = sum(clust_coef.values()) / len(clust_coef)
#     # print("average_clust: ", avg_clust)
#     dotfile_df['average_clustering_coefficient'] = avg_clust
#
#     # average degrees of origami graph
#     N, K = g.order(), g.size()
#     avg_deg = float(K) / N
#     # print("average_degree: ", avg_deg)
#     dotfile_df['average_degree'] = avg_deg
#
#     # betweenness centrality of graph edges
#     bet_cen = nx.betweenness_centrality(g)
#     # print("betweenness_centrality : ", bet_cen)
#     bet_cen_values = sum(list(bet_cen.values())) / len(bet_cen)
#     # print("average bet cen: ", bet_cen_values)
#     dotfile_df['average_betweenness_centrality'] = bet_cen_values
#
#     # closeness centrality of graph
#     clo_cen = nx.closeness_centrality(g)
#     # print("closeness_centrality : ", clo_cen)
#     clo_cen_values = sum(list(clo_cen.values())) / len(clo_cen)
#     # print("average clo cen: ", clo_cen_values)
#     dotfile_df['average_closeness_centrality'] = clo_cen_values
#
#     # top nodes (staples) calculated from centrality measures
#     # sort_bet_cen = sorted(bet_cen, reverse=True)
#     # top_bet_cen = sort_bet_cen[:50]
#     #
#     # sort_clo_cen = sorted(bet_cen, reverse=True)
#     # top_clo_cen = sort_clo_cen[:50]
#     #
#     # print(top_bet_cen)
#     # print(top_clo_cen)
#
#     # calculate the assortivity value of the graph
#     # print("assortivity value: ", nx.algorithms.degree_assortativity_coefficient(g))  # interesting
#     dotfile_df['graph_assortativity'] = nx.algorithms.degree_assortativity_coefficient(g)
#
#     # calculate graph diameter
#     # print("diameter of graph: ", nx.algorithms.distance_measures.diameter(g))
#     dotfile_df['graph_diameter'] = nx.algorithms.distance_measures.diameter(g)
#
#     # calculate graph reciprocity
#     # print("graph reciprocity: ", nx.algorithms.overall_reciprocity(g))
#     dotfile_df['graph_reciprocity'] = nx.algorithms.overall_reciprocity(g)
#
#     # calculate the rich club coefficient (not very good in protein interactions, maybe not good here
#     # print("rich-club coef: ", nx.algorithms.rich_club_coefficient(g))
#
#     """
#     A small world network is characterized by a small average shortest path length, and a large clustering coefficient.
#     Small-worldness is commonly measured with
#     the coefficient sigma or omega.
#     Both coefficients compare the average clustering coefficient and shortest path length of a given graph
#     against the same quantities for an
#     equivalent random or lattice graph.
#
#     NOTE: These seem quite computationally expensive to calculate
#     """
#     # print("small-world-ness sigma: ", nx.algorithms.sigma(g))
#     # print("small-world-ness omega: ", nx.algorithms.omega(g))
#
#     '''
#     https://www.researchgate.net/publication/224366095_Measuring_the_
#     effectiveness_of_the_s-metric_to_produce_better_network_models
#
#     In order to better distinguish between these networks, the metric s was introduced to measure
#     how interconnected the hub nodes are in a network.
#     '''
#     # print("s-metric: ", nx.algorithms.s_metric(g, normalized=''))
#     dotfile_df['s-metric'] = nx.algorithms.s_metric(g, normalized='')
#
#     # https://en.wikipedia.org/wiki/Wiener_index
#     # print("wiener index: ", nx.algorithms.wiener_index(g))
#     dotfile_df['wiener_index'] = nx.algorithms.wiener_index(g)
#
#     # linalg calculations
#     '''https://networkx.github.io/documentation/stable/reference/linalg.html#module-networkx.linalg.modularitymatrix
#     Algebraic connectivity and fiedler vectors of undirected graphs
#     adjacency eigenvalues
#     laplacian eigenvalues
#     bethe hessian eigenvalues
#     normalized laplace eigenvalues
#     modularity spectrum eigenvalues
#     '''
#
#     # print("alg connectivity: ", nx.linalg.algebraicconnectivity.algebraic_connectivity(g))
#     # print("fiedler vector: ", nx.linalg.algebraicconnectivity.fiedler_vector(g))
#     # print("spectral ordering: ", nx.linalg.algebraicconnectivity.spectral_ordering(g))
#     # print("adjacency spectrum: ", nx.linalg.spectrum.adjacency_spectrum(g))
#     # print("laplacian spectrum: ", nx.linalg.spectrum.laplacian_spectrum(g))
#     # print("bethe hessian spectrum: ", nx.linalg.spectrum.bethe_hessian_spectrum(g))
#     # print("norm laplace spectrum: ", nx.linalg.spectrum.normalized_laplacian_spectrum(g))
#     # print("modularity spectrum: ", nx.linalg.spectrum.modularity_spectrum(g))
#
#     print(features_list)
#     store_features_list.append(features_list)
#     dotfile_df.to_csv(cwd + "/" + file_name[1] + "_features.csv")
#     print(dotfile_df.columns)
#     print(dotfile_df.shape)
#     print(dotfile_df)
