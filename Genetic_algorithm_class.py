import numpy as np
import matplotlib.pyplot as plt
from pyeasyga import pyeasyga
import random
import matplotlib.gridspec as gridspec
import copy
from scipy.sparse.linalg import expm
import scipy
from scipy.stats.mstats import gmean
from timeit import default_timer as timer

import re
import os
import config
import time

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()


# Only thing we need to specify below is fitness function.
# ga.fitness_function = customized_fitness_function.  Fitnessfunction should have form : function(genes, seed_data) . Here seed_data is parameters we feed in.
# also need to define your own create_individual and crossover function.
class Extend_Genetic_algorithm(pyeasyga.GeneticAlgorithm):
    def __init__(self,
                 seed_data,
                 population_size=50,
                 generations=100,
                 crossover_probability=0.8,
                 mutation_probability=0.2,
                 immigration_ratio = 0.2,
                 immigrantion_frequency = 0.1,
                 elitism=True,
                 maximise_fitness=True,
                 info_file = None):
        # inherit from parent and define your own variable here in initialization function
        super().__init__(seed_data=seed_data,population_size=population_size,
                         generations=generations, crossover_probability= crossover_probability,
                         mutation_probability=mutation_probability, elitism=elitism,
                         maximise_fitness=maximise_fitness)

        # first list for fitness function. Record all result computed in history
        # Second list for genes. Record all result computed in history.
        self.Biglist_genes = []
        self.Biglist_fitness = []

        # after 0.1 of all generations, different process have to change individuals (some individual immigrate to new place)
        self.immigrate_rate = immigrantion_frequency

        self.immigrate_generation = max(int(self.generations * self.immigrate_rate) , 1)

        self.immigrant_num = int(self.population_size * immigration_ratio)
        self.immigrant_list = []

        self.info_file = info_file

    def run(self):
        """Run (solve) the Genetic Algorithm."""

        start_time = timer()
        self.create_first_generation()
        end_time =  timer()

        if(rank == 0):
            self.info_file.write('time for first generation:  ' + str(end_time - start_time) + '\n')

        for generation_number in range(1, self.generations):
            start_time = timer()

            # Immigrate function
            if self.immigrate_generation != 0 :
                if(generation_number % self.immigrate_generation == 0 ):
                    self.immigrate_population()

            self.create_next_generation()

            end_time = timer()

            if(rank == 0):
                self.info_file.write('time for  generation  ' + str(generation_number) + "  : " + str(end_time - start_time) + '\n')

    def immigrate_population(self):
        '''
        immigrate population between different process.
        :return:
        '''
        immigrant_index_list = []

        # clear immigrant list
        self.immigrant_list = []
        immigrant_num = 0

        if(self.immigrant_num != 0):
            while(immigrant_num < self.immigrant_num):
                immigrant_index  = np.random.choice(self.population_size)
                # we don't want to send same population to other process.
                if immigrant_index not in immigrant_index_list:
                    immigrant = self.current_generation[immigrant_index]
                    self.immigrant_list.append(  copy.deepcopy(immigrant) )

                    immigrant_index_list.append(immigrant_index)
                    immigrant_num = immigrant_num + 1

            # we are going to replace element in immigrant_index from immigrant from other process
            # we have to send bit code and fitness function separately
            immigrant_fitness = np.array([member.fitness for member in  self.immigrant_list])
            immigrant_genes = np.array([member.genes for member in self.immigrant_list ])

            gene_len = len(immigrant_genes[0])

            recv_genes = []
            recv_fitness = []

            # when use MPI_Gather or MPI_recv, make sure the datatype is the same: dtype = np.int 64 != dtype = 'i'
            if(rank == 0):
                recv_genes = np.empty([num_proc, self.immigrant_num , gene_len ] , dtype = np.int64)
                recv_fitness = np.empty([num_proc, self.immigrant_num], dtype = np.float64)

            comm.Gather(immigrant_genes ,recv_genes , 0 )
            comm.Gather(immigrant_fitness, recv_fitness, 0)

            immigrant_genes_recv_shuffle = []
            immigrant_fitness_recv_shuffle = []
            if (rank == 0):
                # reorder the list we receive and send back to each process.
                arr_random = np.arange(num_proc)
                np.random.shuffle(arr_random)

                immigrant_genes_recv_shuffle = np.array([ recv_genes[i] for i in arr_random])
                immigrant_fitness_recv_shuffle =  np.array([recv_fitness[i] for i in arr_random])

            # replace immigrant_fitness and immigrant_genes
            # Notice: when using comm.Scatter, make sure your object: both recev buffer and send_buffer are numpy object
            comm.Scatter(immigrant_genes_recv_shuffle , immigrant_genes , 0)
            comm.Scatter(immigrant_fitness_recv_shuffle, immigrant_fitness, 0 )

            # reconstruct chromosome
            for i in range(self.immigrant_num):
                self.immigrant_list[i].genes = immigrant_genes[i]
                self.immigrant_list[i].fitness = immigrant_fitness[i]

            # now we replace chromosome there with chromosome from other process.
            for i in range( len(immigrant_index_list) ):
                index = immigrant_index_list[i]

                member = self.immigrant_list[i]
                self.current_generation[index].fitness = member.fitness
                self.current_generation[index].genes = copy.deepcopy(member.genes).tolist()



    def create_first_generation(self):
        # We may also read parameter from previous simulation if we set bool_Read_parameter == True
        self.create_initial_population()

        self.calculate_population_fitness()

        self.rank_population()


    def create_next_generation(self):
        """Create subsequent populations, calculate the population fitness and
        rank the population by fitness in the order specified.
        """
        self.create_new_population()
        self.calculate_population_fitness()
        self.rank_population()


    def create_new_population(self):
        """Create a new population using the genetic operators (selection,
               crossover, and mutation) supplied.
               """
        new_population = []
        new_population_gene = []
        elite = copy.deepcopy(self.current_generation[0])
        selection = self.selection_function

        while len(new_population) < self.population_size:
            parent_1 = copy.deepcopy(selection(self.current_generation))

            # parent_2 should be different from parent_1. Same parents are not allowed to mate
            parent_2 = copy.deepcopy(selection(self.current_generation))

            # parent_2 = copy.deepcopy(parent_1)
            # if self.population_size>=2 :
            #     while(parent_2.genes == parent_1.genes):
            #         parent_2 = copy.deepcopy(selection(self.current_generation))


            child_1, child_2 = parent_1, parent_2

            can_crossover = random.random() < self.crossover_probability
            can_mutate = random.random() < self.mutation_probability

            if can_crossover:
                child_1.genes, child_2.genes = self.crossover_function(
                    parent_1.genes, parent_2.genes)

            if can_mutate:
                self.mutate_function(child_1.genes)
                self.mutate_function(child_2.genes)

            # we should not have multiple same genes in one generation, this will make us go to false minimum
            if(not (child_1.genes in new_population_gene) ):
                if (can_mutate or can_crossover):
                    # only re-compute fitness function if there is mutation or crossover
                    child_1.fitness = 0
                new_population.append(child_1)
                new_population_gene.append(child_1.genes)

            if len(new_population) < self.population_size:
                if(not (child_2.genes in new_population_gene)):
                    if(can_mutate or can_crossover):
                        # only re-compute fitness function if there is mutation or crossover
                        child_2.fitness = 0
                    new_population.append(child_2)
                    new_population_gene.append(child_2.genes)

        if self.elitism:
            new_population[0] = elite

        self.current_generation = new_population

    # customized create_individual function and crossover_function.
    def create_individual_new(self):
        # we need to specify the form of seed_data
        _ , full_system_instance, bit_per_parameter, parameter_number = self.seed_data
        return [random.randint(0, 1) for _ in range(parameter_number * bit_per_parameter)]

    def crossover_function(self, parent_1, parent_2):
        # we need to specify the form of seed_data
        _ , full_system_instance, bit_per_parameter, parameter_number = self.seed_data
        # notice because multiple bit represent one data, it makes no sense when do crossover, mess up structure within unit of parameter
        crossover_index = random.randrange(1, parameter_number)
        index = crossover_index * bit_per_parameter
        child_1 = parent_1[:index] + parent_2[index:]
        child_2 = parent_1[index:] + parent_2[:index]

        return child_1, child_2

    def create_initial_population(self):
        Initialization_size = int(self.population_size)
        initial_population = []
        for _ in range(Initialization_size):
            genes = self.create_individual_new()
            individual = pyeasyga.Chromosome(genes)
            initial_population.append(individual)
        self.current_generation = initial_population



    def calculate_population_fitness(self):
        """Calculate the fitness of every member of the given population using
        the supplied fitness_function. We only compute mutated or crossover genes.
        """
        for individual in self.current_generation:
            # if fitness == 0. that means this is newly generated child or parents. We have to re-compute its fitness function.
            if individual.fitness == 0:
                # search if this genes is already in Biglist_genes list. This approach is taken only when computation of cost function is extremley expensive
                if( individual.genes not in self.Biglist_genes ):
                    individual.fitness = self.fitness_function(
                        individual.genes, self.seed_data)
                    self.Biglist_genes.append(individual.genes)
                    self.Biglist_fitness.append(individual.fitness)
                else:
                    # already computed in the history. Record its value
                    index = self.Biglist_genes.index(individual.genes)
                    individual.fitness = self.Biglist_fitness[index]




