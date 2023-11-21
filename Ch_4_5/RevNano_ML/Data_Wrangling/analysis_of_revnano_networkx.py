import pygraphviz as pgv
from pygraphviz import *
import networkx as nx
from networkx import *

"""
Assuming all origami's are produced in a similar representation to the 
test file: rothemund_rect_triangle
We should be able to calculate around 17 features from """
revnano_output_path = "/media/jordan/PhD-Backup/Back_Up_Folder/Data_Set_Rev_Nano/ben_s_e-revnano/_mid/"
origami_name = "rothemund_rect_triangle"
dot_file_name = revnano_output_path + origami_name + "_3_dotfile.dot"
print(dot_file_name)

# Import using pygraphviz
G = pgv.AGraph(dot_file_name)

# Swap over to networkx so it doesn't break
g = nx.Graph(G)

# general info of origami graph
print("nodes: ", nx.number_of_nodes(g))
print("edges: ", nx.number_of_edges(g))
avg_neighbour_deg = nx.average_neighbor_degree(g)
avg_neighbour_total = sum(list(avg_neighbour_deg.values())) / len(avg_neighbour_deg)
print("avg neigh: ", avg_neighbour_total)
# density of the graph
print("density of graph: ", nx.density(g))

# transitivity of graph
triadic_closure = nx.transitivity(g)
print("Triadic closure:", triadic_closure)

# Centrality of Staples origami (density is the average of this)
print("deg centrality: ", nx.algorithms.degree_centrality(g))
# Average Connectivity of Staples origami
print("avg deg connectivity: ", nx.average_degree_connectivity(g))
# Average shortest path length of origami
print("avg shortpath: ", nx.average_shortest_path_length(g))

origami_ud = g.to_undirected()
# clustering coefficient of origami graph
clust_coef = nx.clustering(origami_ud)
# average clustering coefficient of origami graph
avg_clust = sum(clust_coef.values()) / len(clust_coef)
print("average_clust: ", avg_clust)

# average degrees of origami graph
N, K = g.order(), g.size()
avg_deg = float(K) / N
print("average_degree: ", avg_deg)

# betweenness centrality of graph edges
bet_cen = nx.betweenness_centrality(g)
print("betweenness_centrality : ", bet_cen)
bet_cen_values = sum(list(bet_cen.values())) / len(bet_cen)
print("average bet cen: ", bet_cen_values)
# closeness centrality of graph
clo_cen = nx.closeness_centrality(g)
print("closeness_centrality : ", clo_cen)
clo_cen_values = sum(list(clo_cen.values())) / len(clo_cen)
print("average clo cen: ", clo_cen_values)

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
print("assortivity value: ", nx.algorithms.degree_assortativity_coefficient(g))  # interesting
# calculate graph diameter
print("diameter of graph: ", nx.algorithms.distance_measures.diameter(g))
# calculate graph reciprocity
print("graph reciprocity: ", nx.algorithms.overall_reciprocity(g))
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
print("s-metric: ", nx.algorithms.s_metric(g, normalized=''))

# https://en.wikipedia.org/wiki/Wiener_index
print("Weiner index: ", nx.algorithms.wiener_index(g))

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
