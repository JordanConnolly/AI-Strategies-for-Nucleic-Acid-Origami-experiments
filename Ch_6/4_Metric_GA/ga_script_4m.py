import array
import os
import random
import json
import numpy
from math import sqrt
import time
import pickle
import glob

# deap code
from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from scoop import futures  # <-------------------- import futures module from scoop

# ben's heuristic code
import jordan_extended
import iohandler
import metrics
import dna
import contactmap.iohandler as cio

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
import itertools as IT

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import multiprocessing
from pathlib import Path

# get config file for genetic algorithm parameters
import json
import re

hyper_parameters_dict = {}
with open('genetic_algorithm_config_1torelli_12cxpb90mut16indmut10.json') as json_file:
    parsed = json.load(json_file)
    # print(json.dumps(parsed, indent=4, sort_keys=True))
    for i in parsed.get("ga_hyper_params"):
        hyper_parameters_dict = i
    json_file.close()

list_of_hyper_params = ["id", "description", "origami_contactmap_filename", "storage_path", "MUTATE_INDIVIDUAL_PB",
                        "MUTATE_ATTR_PB", "MUTATE_BP", "NGEN", "MU", "CXPB"]

# useful tools:
# https://deap.readthedocs.io/en/master/api/tools.html


def ordinal_encoder(data_to_encode):
    my_array = np.array(list(data_to_encode))
    label_encoder = LabelEncoder()
    label_encoder.fit(np.array(['A', 'C', 'G', 'U']))
    integer_encoded = label_encoder.transform(my_array)
    return integer_encoded


def ordinal_decoder(data_to_decode):
    array_to_decode = np.array(list(data_to_decode))
    decoded_as_int = array_to_decode.astype(int)
    decoded = decoded_as_int.astype(str)
    decoded[decoded == "0"] = "A"  # A
    decoded[decoded == "1"] = "C"  # C
    decoded[decoded == "2"] = "G"  # G
    decoded[decoded == "3"] = "U"  # T
    decoded = "".join(decoded)
    return decoded


# time the script
start = time.time()

# import functions from Ben's Scoring Criteria Work
score_scaffold = jordan_extended.score_scaffold  # custom evaluator
generate_random_scaffold = jordan_extended.generate_random_scaffold  # for initialisation (generator)
generate_staples_from_scaffold = jordan_extended.generate_staples_from_scaffold  # for initialisation (generator)
precompute_lookups = jordan_extended.precompute_lookups  # for initialisation (generator)

# contactmap for the origami we are using, placed in the _input/ directory
origami_contactmap_filename = hyper_parameters_dict.get("origami_contactmap_filename")

store_seed = []

# name storage location and experiment name
origami_used = origami_contactmap_filename.split(".")[0]
descriptor_storage_path = hyper_parameters_dict.get("storage_path")
storage_path = "./output_GA/" + origami_used + descriptor_storage_path

metadata_path = storage_path + "metadata/"
results_path = storage_path + "results/"
plot_path = storage_path + "plots/"
checkpoint_path = storage_path + "checkpoints/"

# store the hyper-parameters locally
with open(storage_path + 'genetic_algorithm_config_used.json', "w") as json_file:
    json_file.write(json.dumps(hyper_parameters_dict, indent=4, sort_keys=False))
    json_file.close()

#  create new paths for origami used
# if not os.path.exists(storage_path):
#     os.makedirs(storage_path)
# if not os.path.exists(metadata_path):
#     os.makedirs(metadata_path)
# if not os.path.exists(results_path):
#     os.makedirs(results_path)
# if not os.path.exists(plot_path):
#     os.makedirs(plot_path)

# which rep is performed
script = Path(__file__).stem
rep = int(script.split("_")[::-1][0])

# which seed is chosen (starts from list[1])
random_seed_list = [8738, 5995, 9355, 9598, 2715, 5453, 6287, 5896, 5041, 6416, 3358, 5979, 1791, 6972, 6533, 1071,
                    8281, 5704, 6613, 6609, 5093, 293, 5787, 9740, 1943, 7552, 4037, 2806, 5141, 9906, 8738]

# set experiment name
experiment_name = origami_used + "_GA-experiment_rep_" + str(rep)

# set random seed for experiment
set_seed = random_seed_list[rep]  # set random seed
random.seed(set_seed)  # set it for this script
numpy.random.seed(set_seed)
seed_list = [random.randrange(1, 10000, 1) for _ in range(1000)]  # get random seeds for random scaffold generation

# 1a. load the JSON settings
settings_default = iohandler.parse_json("settings_default_rna.json")
settings = iohandler.convert_json_data(settings_default, settings_default.copy())

# 1b. load and parse contact map file for the origami
# 		(this is needed: it says how the scaffold is connected to the staples)
st, sc, scaffold_circular = cio.read_origami_contactmap("_input/%s" % origami_contactmap_filename)

# 1c. make origami dict, containing all info
origami = {}

origami["contactmap"] = {}
origami["contactmap"]["st"] = st
origami["contactmap"]["sc"] = sc
origami["contactmap"]["scaffold_circular"] = ""
origami["contactmap"]["scaffold_type"] = "LINEAR"
if scaffold_circular : origami["contactmap"]["scaffold_type"] = "CIRCULAR"

origami["settings"] = settings
origami["precomputed"] = precompute_lookups(origami)


def create_origami_instances(seed_chosen):
    # generates origami instance
    # step a. generate a random scaffold sequence (or supply one of your own
    scaffold53 = generate_random_scaffold(origami, seed_chosen)
    # step a. extended: encode scaffold into numeric values
    scaffold53_encoded = ordinal_encoder(scaffold53)
    return scaffold53_encoded


def custom_evaluation(individual):
    # un-encode scaffold
    scaffold53 = ordinal_decoder(individual[0:])
    # step b. derive staples from individuals scaffold
    staple_set = generate_staples_from_scaffold(origami, scaffold53)
    scores = score_scaffold(origami, scaffold53, staple_set)
    print(scores)
    return scores[0], scores[1], scores[2], scores[3]


# Problem definition
# Functions zdt1, zdt2, zdt3, zdt6 have bounds [0, 1], origami_heuristic set to [0, 3] (DNA individuals range)
BOUND_LOW, BOUND_UP = 0.0, 3.0
# Custom BP Segment Mutation Parameters
MUTATE_INDPB = hyper_parameters_dict.get("MUTATE_ATTR_PB")  # probability of mutating each NT of segment (set to 100%)
segment = hyper_parameters_dict.get("MUTATE_BP")  # base pair segment length to mutate

creator.create("FitnessMin_", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0,))
creator.create("Individual_", array.array, typecode='d', fitness=creator.FitnessMin_)
toolbox = base.Toolbox()
toolbox.register("map", futures.map)  # <--------------- overload the map function
toolbox.register("attr_item", create_origami_instances)

toolbox.register("evaluate", custom_evaluation)
toolbox.register("mutation", tools.mutUniformInt, low=BOUND_LOW, up=BOUND_UP, indpb=MUTATE_INDPB)
toolbox.register("select", tools.selNSGA2)


def cxTwelvePoint(ind1, ind2):
    """Executes a hard coded four-point crossover on the input :term:`sequence`
    individuals.
    The two individuals are modified in place and both keep
    their original length.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.randint` function from the Python
    base :mod:`random` module.
    """
    # finds size of inds
    size = min(len(ind1), len(ind2))
    # chooses random points
    cx_point_list = []
    for i in range(12):
        cx_point = random.randint(1, size)
        cx_point_list.append(cx_point)
    cx_point_list.sort()
    # hard code the random points for application of crossovers
    cxpoint1 = cx_point_list[0]
    cxpoint2 = cx_point_list[1]
    cxpoint3 = cx_point_list[2]
    cxpoint4 = cx_point_list[3]
    cxpoint5 = cx_point_list[4]
    cxpoint6 = cx_point_list[5]
    cxpoint7 = cx_point_list[6]
    cxpoint8 = cx_point_list[7]
    cxpoint9 = cx_point_list[8]
    cxpoint10 = cx_point_list[9]
    cxpoint11 = cx_point_list[10]
    cxpoint12 = cx_point_list[11]

    # apply two point cross-over
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    # apply two point cross-over again (4 point crossover)
    ind1[cxpoint3:cxpoint4], ind2[cxpoint3:cxpoint4] \
        = ind2[cxpoint3:cxpoint4], ind1[cxpoint3:cxpoint4]
    # apply two point cross-over again (6 point crossover)
    ind1[cxpoint5:cxpoint6], ind2[cxpoint5:cxpoint6] \
        = ind2[cxpoint5:cxpoint6], ind1[cxpoint5:cxpoint6]
    # apply two point cross-over again (8 point crossover)
    ind1[cxpoint7:cxpoint8], ind2[cxpoint7:cxpoint8] \
        = ind2[cxpoint7:cxpoint8], ind1[cxpoint7:cxpoint8]
    # apply two point cross-over again (10 point crossover)
    ind1[cxpoint9:cxpoint10], ind2[cxpoint9:cxpoint10] \
        = ind2[cxpoint9:cxpoint10], ind1[cxpoint9:cxpoint10]
    # apply two point cross-over again (12 point crossover)
    ind1[cxpoint11:cxpoint12], ind2[cxpoint11:cxpoint12] \
        = ind2[cxpoint11:cxpoint12], ind1[cxpoint11:cxpoint12]
    # return twelve point crossover individuals

    return ind1, ind2


# custom mate
toolbox.register("mate", cxTwelvePoint)


def custom_staple_segment_mutation(inds, segment_bp):
    # turn tuples into list
    inds = list(inds)
    # calculate the position of low and high sections (pointer 1 and 2)
    total_len_inds = round(len(inds))
    useful_len_inds = round(len(inds) - segment_bp)  # ensure there are enough BP to apply mutation
    low_pointer = random.randint(0, useful_len_inds)
    high_pointer = low_pointer + segment_bp
    # select only one section of sequence to optimise
    ind_to_mutate = inds[low_pointer:high_pointer]
    # mutate them
    toolbox.mutation(ind_to_mutate)
    # re-combine the ind1 and ind2 sequences
    ind_mutated = inds[:low_pointer:] + list(ind_to_mutate) + inds[high_pointer:]
    return ind_mutated


# custom mutation
toolbox.register("mutate", custom_staple_segment_mutation)

import pickle

# set hyper-parameters for the NSGA-II
NGEN = hyper_parameters_dict.get("NGEN")  # how many generations to produce
MU = hyper_parameters_dict.get("MU")  # the size of the population (I.E: 100 scaffolds)
CXPB = hyper_parameters_dict.get("CXPB")  # cross-over mutation probability
IND_MU_PB = hyper_parameters_dict.get("MUTATE_INDIVIDUAL_PB")  # probability of choosing ind to mutate

# store seeds used (only implement if running repetitions)
with open(metadata_path + experiment_name + '_seeds_set.txt', 'a') as file:
    store_seed_string = ("rep:" + str(rep) + " seed:" + str(set_seed))
    file.write(store_seed_string + '\n')

####### checkpoint code #########
# search pathway to files using glob.glob checkpoint pathway


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


checkpoint_pathway_names = sorted(glob.glob(checkpoint_path + "*.pickle"), key=numerical_sort)

print(checkpoint_pathway_names)
# set up the checkpoint variable
if not checkpoint_pathway_names:
    checkpoint_variable = None

if checkpoint_pathway_names:
    # automate the running of checkpoint
    # find the pathway that corresponds to seed
    checkpoint_files_of_this_seed = []
    for checkpoints in checkpoint_pathway_names:
        checkpoint_pathway_split = checkpoints.split("/")
        checkpoint_pickle_file = checkpoint_pathway_split[-1]
        checkpoint_file = checkpoint_pickle_file.split(".")[0]
        checkpoint_file_seed = checkpoint_file.split("_")[0]
        if checkpoint_file_seed == str(set_seed):
            checkpoint_files_of_this_seed.append(checkpoints)

    # find the max generation produced by pickle
    print(checkpoint_files_of_this_seed)
    max_checkpoint_generation = 0
    updated_checkpoint_file_pathway = None
    for checkpoints in checkpoint_files_of_this_seed:
        checkpoint_pathway_split = checkpoints.split("/")
        checkpoint_pickle_file = checkpoint_pathway_split[-1]
        checkpoint_file = checkpoint_pickle_file.split(".")[0]
        print(checkpoint_file.split("_")[1])
        if int(checkpoint_file.split("_")[1]) > max_checkpoint_generation:
            max_checkpoint_generation = int(checkpoint_file.split("_")[1])
            updated_checkpoint_file_pathway = checkpoints
    # find the pathway that corresponds to both seed and max
    checkpoint_variable = updated_checkpoint_file_pathway


print(checkpoint_variable)


def main(seed=set_seed, checkpoint=checkpoint_variable):
    pareto = tools.ParetoFront()
    # add stats tools to be logged
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    if checkpoint:
        # use a random seed for reproducibility of work
        random.seed(seed)

        # A file name has been given, then load the data from the file
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        pop = cp["population"]
        start_gen = cp["generation"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
        scores = cp["scores"]
    else:
        # use a random seed for reproducibility of work
        random.seed(seed)

        scores = []
	
        # create the population
        import itertools as IT
        counter = IT.count(1)
        l = range(MU)
        # every time we iterate in the list comprehension, increase the counter and use a new seed for scaffold generation
        pop = [creator.Individual_(toolbox.attr_item(seed_list[next(counter)])) for i, x in enumerate(l, start=1)]
        print(len(pop))
        print(ordinal_decoder(pop[0]))  # DNA sequence of one individual

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            scores.append(ind.fitness.values)

        # This is just to assign the crowding distance to the individuals; no actual selection is done
        pop = toolbox.select(pop, len(pop))

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)
        gen_0_time = time.time() - start

        # store single_generation time
        with open(metadata_path + experiment_name + '_gen_' + 'time.txt', 'a') as file:
            store_first_gen_time_string = ("Gen_" + str(0) + "_Time:" + str(gen_0_time))
            file.write(store_first_gen_time_string + '\n')

        # store Gen 0 meta-data
        gen_0_pop = [ordinal_decoder(gen_0_scaffold) for gen_0_scaffold in pop]
        df_pop_gen_0 = pd.DataFrame(gen_0_pop)
        # saving the dataframes
        df_pop_gen_0.to_csv(metadata_path + experiment_name + "_scaffolds_gen_0" + '.csv')

        # Create generational process terminator counters
        fits = [ind.fitness.values for ind in pop]
        df_pop_fit_0 = pd.DataFrame(fits)
        df_pop_fit_0.to_csv(metadata_path + experiment_name + "_fits_gen_0" + '.csv')
        # early_stop_count = 1
        start_gen = 0

    gen = 0 + start_gen

    for gen in range(start_gen, NGEN):
        gen += 1
        print("current gen:", gen)
        # print("current early-stop counter:", early_stop_count)
        # prior_fits = min(fits)

        # collect the time of generation
        gen_start = time.time()

        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                # print(ind1)
                # print(ind2)
                toolbox.mate(ind1, ind2)

            if random.random() <= IND_MU_PB:
                toolbox.mutate(ind1, segment)
                toolbox.mutate(ind2, segment)
                # print("mutated ind1", ind1)
                # print("mutated ind2", ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            scores.append(ind.fitness.values)

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        gen_time = time.time() - gen_start

        # store single_generation time
        with open(metadata_path + experiment_name + "_rep_" + str(rep) + '_gen_' + 'time.txt', 'a') as file:
            store_gen_time_string = ("Gen_" + str(gen) + "Time:" + str(gen_time))
            file.write(store_gen_time_string + '\n')

        # store main_gen meta-data
        gen_pop = [ordinal_decoder(gen_scaffold) for gen_scaffold in pop]
        df_pop_gen = pd.DataFrame(gen_pop)
        df_stats_gen = pd.DataFrame(logbook)
        # saving the dataframes
        df_pop_gen.to_csv(metadata_path + experiment_name + "_scaffolds_gen_" + str(gen) + '.csv')
        df_stats_gen.to_csv(metadata_path + experiment_name + "_logbook_gen.csv")  # in case of early termination
        # check metric 1 to see if it has changed
        
        fits = [ind.fitness.values for ind in pop]
        df_fits = pd.DataFrame(fits)
        df_fits.to_csv(results_path + experiment_name + "_fits_gen_" + str(gen) + '.csv')

        # if min(fits) != prior_fits:
        #     early_stop_count = 0
        # if min(fits) == prior_fits:
        #     early_stop_count += 1
        total_time = time.time() - start
        # store seeds used (only implement if running repetitions)
        with open(metadata_path + experiment_name + '_total_run_time.txt', 'a') as file:
            store_total_time_string = ("Total Time:" + str(total_time))
            file.write(store_total_time_string + '\n')

        pareto.update(pop)

        print(gen)

        # checkpoint pickle
        if gen % 50 == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=pop, generation=gen,
                      logbook=logbook, rndstate=random.getstate(), scores=scores)
            with open(checkpoint_path + str(set_seed) + "_" + str(gen) + "_checkpoint.pickle", "wb") as cp_file:
                pickle.dump(cp, cp_file)

    return pop, logbook, pareto, scores


if __name__ == "__main__":

    pop, stats, optimal_front, scores = main()

    # store Final Gen results
    final_gen_pop = [ordinal_decoder(final_gen_scaffold) for final_gen_scaffold in pop]
    df_final_gen_pop = pd.DataFrame(final_gen_pop)
    df_final_gen_stats = pd.DataFrame(stats)
    df_final_gen_graph = pd.DataFrame(scores)
    # saving the dataframes
    df_final_gen_pop.to_csv(results_path + experiment_name + "_scaffolds_final_gen.csv")
    df_final_gen_stats.to_csv(results_path + experiment_name + "_logbook_final_gen.csv")
    df_final_gen_graph.to_csv(results_path + experiment_name + "_scores_final_gen.csv")

    # Saving the Pareto Front, for further exploitation
    with open(results_path + experiment_name + '_pareto_front.txt', 'w') as front:
        for ind in optimal_front:
            front.write(str(ind.fitness) + '\n')
    print(stats)
