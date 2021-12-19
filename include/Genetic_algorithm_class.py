import numpy as np
from pyeasyga import pyeasyga
import random
import copy
from timeit import default_timer as timer
from include.util import shuffle_data

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()



# Need to specify fitness function for Genetic algorithm module
# ga.fitness_function = customized_fitness_function.  Fitnessfunction should have form : function(genes, seed_param) . Here seed_param is parameters we feed in.
# You have to define your own create_individual and crossover function.
# This is continuous genetic algorithm, which means paramter to be optimized is continuous. See: Section 3 of Practical Genetic algorithms (Second Edition) -- Randy L. Haupt, Sue Ellen Haupt.
# class below inherit from pyeasyga class : see https://pypi.org/project/pyeasyga/
class Extend_Genetic_algorithm(pyeasyga.GeneticAlgorithm):
    def __init__(self,
                 seed_data,
                 population_size=50,
                 generations=100,
                 crossover_probability=0.8,
                 mutation_probability=0.2,
                 immigration_ratio = 0.2,
                 immigrantion_rate = 0.1,
                 elitism=True,
                 maximise_fitness=True,
                 info_file = None):
        # inherit from parent and define your own variable here in initialization function
        super().__init__(seed_data=seed_data,population_size=population_size,
                         generations=generations, crossover_probability= crossover_probability,
                         mutation_probability=mutation_probability, elitism=elitism,
                         maximise_fitness=maximise_fitness)

        #  Record fitness function and genes we have computed before in list to avoid duplicate computation.
        self.Biglist_genes = []
        self.Biglist_fitness = []

        # after 0.1 of all generations, different process have to change individuals (some individual immigrate to other process )
        self.immigrate_rate = immigrantion_rate
        # immigrate_generation : do immigration if generation_num % immigrate_generation == 0.
        self.immigrate_generation = max(int(self.generations * self.immigrate_rate) , 1)
        # number of species to immigrate
        self.immigrant_num = int(self.population_size * immigration_ratio)
        self.immigrant_list = []

        self.info_file = info_file

        # number and range of parameter to optimize.
        param_range , _ ,  param_number = self.seed_data
        self.param_range = param_range
        self.param_number = param_number


    def run(self):
        """Run the Genetic Algorithm."""

        # create first generation.
        start_time = timer()
        self.create_first_generation()
        end_time =  timer()

        # output time for simulation of one generation. unit : seconds
        if(rank == 0):
            self.info_file.write('time for first generation:  ' + str(end_time - start_time) + 's \n')

        for generation_number in range(1, self.generations):
            start_time = timer()

            # Immigrate function
            if(generation_number % self.immigrate_generation == 0 ):
                self.immigrate_population()

            self.create_next_generation()

            end_time = timer()

            if(rank == 0):
                self.info_file.write('time for  generation  ' + str(generation_number) + "  : " + str(end_time - start_time) + 's \n')



    def create_first_generation(self):
        # We may also read parameter from previous simulation if we set bool_Read_parameter == True
        self.create_initial_population()
        # compute fitness function for individuals in population ensemble
        self.calculate_population_fitness()
        # rank individual according to fitness function .
        self.rank_population()


    def create_next_generation(self):
        """Create subsequent populations, calculate the population fitness and
        rank the population by fitness in the order specified.
        """
        # do mutation and crossover to generate new population.
        self.create_new_population()
        # compute fitness function for individuals in population ensemble
        self.calculate_population_fitness()
        # rank individual according to fitness function .
        self.rank_population()

    def create_initial_population(self):
        Initialization_size = int(self.population_size)
        self.current_generation = []
        for _ in range(Initialization_size):
            genes = self.create_individual_gene()
            individual = pyeasyga.Chromosome(genes)
            self.current_generation.append(individual)

    def create_new_population(self):
        """    Create a new population using the genetic operators (selection,
               crossover, and mutation) supplied.
               """

        def append_individual_in_new_population(individual , new_population, new_population_gene , can_mutate, can_crossover):
            # we should not have  same genes in one generation, this may lead us go to local minimum
            if(not(individual.genes in new_population_gene)):
                if( can_mutate or can_crossover):
                    # only re-compute fitness function if there is mutation or crossover.
                    # in program, we will only recompute fitness function when it is 0.
                    individual.fitness = 0

                new_population.append(individual)
                new_population_gene.append(individual.genes)

        new_population = []
        new_population_gene = []
        # individual with largest fitness function
        elite = copy.deepcopy(self.current_generation[0])
        selection = self.selection_function

        while len(new_population) < self.population_size:
            parent_1 = copy.deepcopy(selection(self.current_generation))

            parent_2 = copy.deepcopy(selection(self.current_generation))

            child_1, child_2 = parent_1, parent_2

            can_crossover = random.random() < self.crossover_probability
            can_mutate = random.random() < self.mutation_probability

            if can_crossover:
                child_1.genes, child_2.genes = self.crossover_function(
                    parent_1.genes, parent_2.genes)

            if can_mutate:
                self.mutate_function(child_1.genes)
                self.mutate_function(child_2.genes)

            append_individual_in_new_population(child_1,  new_population, new_population_gene , can_mutate, can_crossover)

            if len(new_population) < self.population_size:
                append_individual_in_new_population(child_2, new_population, new_population_gene , can_mutate, can_crossover )

        if self.elitism:
            new_population[0] = elite

        self.current_generation = new_population

    # -------------- customized create_individual function , mutation_function and crossover_function.  -----------------
    def crossover_function(self, parent_1, parent_2):
        '''
        cross over function for continuous variable.
        :param parent_1:
        :param parent_2:
        :return:
        '''
        # rn in range [0,1]
        rn = np.random.random([self.param_number])
        child_1 = parent_1 - rn * (parent_1 - parent_2)
        child_2 = parent_2 + rn * (parent_1 - parent_2)

        return child_1, child_2

    def mutate_function(self, gene):
        '''
        mutate function for continuous variable. change one variable to random number
        :param gene: gene to do mutation
        :return:
        '''
        rn = random.random()
        mutate_index = random.randrange(self.param_number)
        gene[mutate_index] = rn * self.param_range

    def create_individual_gene(self):
        return [ np.random.random( self.param_number ) * self.param_range  ]

#  --------------------------------------------------------------------------------------------

    def calculate_population_fitness(self):
        """Calculate the fitness of every member of the given population using
        the supplied fitness_function.
         We only compute fitness function for mutated and crossovered genes, which is 0.
        """
        for individual in self.current_generation:
            # if fitness == 0. that means this is newly generated child or parents. We have to compute its fitness function.
            if individual.fitness == 0:
                # search if this genes is already in Biglist_genes list. This approach is taken only when computation of cost function is extremley expensive
                if( individual.genes not in self.Biglist_genes ):
                    # fitness function is main computation part:  We run SUR algorithm and analyze P(t) results.
                    individual.fitness = self.fitness_function(
                        individual.genes, self.seed_data)
                    self.Biglist_genes.append(individual.genes)
                    self.Biglist_fitness.append(individual.fitness)
                else:
                    # already computed in the history. Record its value
                    index = self.Biglist_genes.index(individual.genes)
                    individual.fitness = self.Biglist_fitness[index]

# ----------------------- immigration module ------------------------------------

    def shuffle_genetic_data(self , immigrant_index_list ):
        '''
        immigrant_index_list : index for individual in list to immigrate.
        used for immigration.
        :return:
        '''
        immigrant_fitness = np.array([member.fitness for member in self.immigrant_list])
        immigrant_genes = np.array([member.genes for member in self.immigrant_list])

        arr_random = []
        if(rank == 0):
            arr_random = np.arange(num_proc)
            np.random.shuffle(arr_random)

        immigrant_fitness = shuffle_data(immigrant_fitness , num_proc, arr_random)
        immigrant_genes = shuffle_data(immigrant_genes , num_proc, arr_random)

        # reconstruct chromosome
        for i in range(self.immigrant_num):
            self.immigrant_list[i].genes = immigrant_genes[i]
            self.immigrant_list[i].fitness = immigrant_fitness[i]

        # now we replace native chromosome with immigrant chromosome from other process
        for i in range(len(immigrant_index_list)):
            index = immigrant_index_list[i]

            member = self.immigrant_list[i]
            self.current_generation[index].fitness = member.fitness
            self.current_generation[index].genes = copy.deepcopy(member.genes).tolist()

    def immigrate_population(self):
        '''
        immigrate population between different process for better genetic algorithm performance.
        :return:
        '''

        # clear immigrant list
        immigrant_index_list = []
        self.immigrant_list = []
        current_immigrant_num = 0

        if(self.immigrant_num != 0):
            while(current_immigrant_num < self.immigrant_num):
                immigrant_index  = np.random.choice(self.population_size)
                # we don't want to send same population to other process.
                if immigrant_index not in immigrant_index_list:
                    immigrant_index_list.append(immigrant_index)
                    current_immigrant_num = current_immigrant_num + 1

                    immigrant = self.current_generation[immigrant_index]
                    self.immigrant_list.append(  copy.deepcopy(immigrant) )

            # shuffle genetic data and replace native chromosome with immigrant chromosome.
            self.shuffle_genetic_data(immigrant_index_list)


# --------------- immigration module ------------------------------------



