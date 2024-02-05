from pyeasyga import pyeasyga
from timeit import default_timer as timer
from include.util import rank, num_proc, shuffle_data
import copy
import random
import numpy as np
import src.simulate_energy_flow_between_photon_and_detector

'''
Need to specify fitness function for Genetic algorithm module
ga.fitness_function = customized_fitness_function.  Fitnessfunction should have form : function(genes, seed_param) . Here seed_param is parameters we feed in.
You have to define your own create_individual and crossover function.
This code uses continuous genetic algorithm, which means parameter to be optimized is continuous. See: Section 3 of Practical Genetic algorithms (Second Edition) -- Randy L. Haupt, Sue Ellen Haupt.
class below inherit from pyeasyga class : see https://pypi.org/project/pyeasyga/
'''

class ExtendGeneticAlgorithm(pyeasyga.GeneticAlgorithm):
    '''
    Extended Genetic algorithm class.
    This class is defined as inheritance of pyeasyga.GeneticAlgorithm class (https://pyeasyga.readthedocs.io/en/latest/)

    '''
    # import method.

    def __init__(self,
                 seed_data,
                 population_size=50,
                 generations=100,
                 crossover_probability=0.8,
                 mutation_probability=0.2,
                 immigration_population_ratio = 0.2,
                 immigrantion_rate = 0.1,
                 elitism=True,
                 maximise_fitness=True,
                 info_file = None):
        '''
        For fitness function:  fitness_function (gene, seed_data)

        :param seed_data: input data to the Genetic Algorithm
        seed_data = [coupling_parameter_range, full_system_instance, parameter_number, highest_peak_bool]
        :type seed_data: list of objects
        :param population_size: size of population
        :param generations: number of generations to evolve
        :param crossover_probability:  probability of crossover operation
        :param mutation_probability: probability of mutation operation
        :param immigration_population_ratio: ratio of population for immigration.
        :param immigrantion_rate: rate / frequency of immigration.
        :param elitism: bool. if true, the best individual is not disgarded.
        :param maximise_fitness: bool. If true, rank the individual by descending order according to fitness value.
        :param info_file:

        '''
        # inherit from parent and define your own variable here in initialization function
        super().__init__(seed_data=seed_data,population_size=population_size,
                         generations=generations, crossover_probability= crossover_probability,
                         mutation_probability=mutation_probability, elitism=elitism,
                         maximise_fitness=maximise_fitness)

        #  Record fitness function and genes we have computed before in list to
        #  avoid duplicate computation of the fitness function.
        # this is because evaluating fitness function could be expensive.
        self.previous_genes = []
        self.previous_fitness = []

        # different process have to change individuals (some individual immigrate to other process )
        self.immigrate_rate = immigrantion_rate

        # immigrate_generation : do immigration if generation_num % immigrate_generation == 0.
        self.immigrate_generation = max(int(self.generations * self.immigrate_rate) , 1)

        self.immigrant_num = int(self.population_size * immigration_population_ratio)  # number of species to immigrate

        self.immigrant_list = []  # list for individual to immigrate to other processes

        self.info_file = info_file


        # number and range of parameter to optimize.
        parameter_range , _ ,  parameter_number , _ = self.seed_data
        self.parameter_range = parameter_range
        self.parameter_number = parameter_number

        self.fitness_function = src.simulate_energy_flow_between_photon_and_detector.fitness_function_for_individual_full_system_instance


    def run(self):
        """Run the Genetic Algorithm."""
        # Create first generation.
        start_time = timer()
        self.create_first_generation()
        end_time =  timer()

        # Output time for simulation of one generation. unit : seconds
        if rank == 0:
            self.info_file.write('time for first generation:  ' + str(end_time - start_time) + 's \n')

        for generation_number in range(1, self.generations):
            start_time = timer()

            # Immigrate function
            if generation_number % self.immigrate_generation == 0:
                self.immigrate_population()

            self.create_next_generation()

            end_time = timer()

            if rank == 0:
                self.info_file.write('time for  generation  ' + str(generation_number) + "  : " + str(end_time - start_time) + 's \n')


    def show_current_generation(self):
        return self.current_generation.copy()



    '''
    code to create first generation and next generation for genetic algorithm.
    '''

    def create_first_generation(self):
        '''
        create first generation and evaluate the fitness function for genetic algorithm.
        :param self:
        :return:
        '''
        # We may also read parameter from previous simulation if we set bool_Read_parameter == True
        self.create_initial_population()
        # compute fitness function for individuals in population ensemble
        self.calculate_population_fitness()
        # rank individual according to fitness function .
        self.rank_population()

    def create_next_generation(self):
        """Create subsequent populations, calculate the population fitness and
        rank the population by fitness in descending order.
        """
        # do mutation and crossover to generate new population.
        self.create_new_population()
        # compute fitness function for individuals in population ensemble
        self.calculate_population_fitness()
        # rank individual according to fitness function .
        self.rank_population()

    def create_initial_population(self):
        '''
        create the initial population for the genetic algorithm.
        calculate the population fitness and rank the population by fitness in descending order.
        :param self:
        :return:
        '''
        initial_population_size = int(self.population_size)
        self.current_generation = []
        for _ in range(initial_population_size):
            genes = self.create_individual_gene()
            individual = pyeasyga.Chromosome(genes)  # (fitness, genes)
            self.current_generation.append(individual)

    def create_new_population(self):
        """
        Create a new population using the genetic operators (selection,
               crossover, and mutation) supplied.
        """

        def append_individual_in_new_population(individual, new_individual_list1, new_individual_gene_list1, mutate_bool,
                                                crossover_bool):
            '''

            :param individual: individual in population for genetic algorithm
            :type: individual: chromosome
            :param new_individual_list1: a list of new individual chromosome
            :param new_individual_gene_list1: The gene of new individuals
            :param mutate_bool: if True, mutate the genes.
            :param crossover_bool: if True, do crossover to genes
            :return:
            '''
            # we should not have the same genes in one generation, this may lead us go to local minimum
            if not (individual.genes in new_individual_gene_list1):
                if mutate_bool or crossover_bool:
                    # only re-compute fitness function if there is a need for mutation or crossover.
                    # in the program, we will only recompute fitness function when it is 0.
                    individual.fitness = 0

                new_individual_list1.append(individual)
                new_individual_gene_list1.append(individual.genes)



        new_individual_list = []  # list of individual chromosome for new generation.
        new_individual_gene_list = []  # list of individual chromosome's gene for new generation.
        # individual with the largest fitness function
        elite = copy.deepcopy(self.current_generation[0])  # the individual with the largest fitness function.
        selection = self.selection_function

        while len(new_individual_list) < self.population_size:
            # choose two parent individual for crossover or mutation.
            parent_1 = copy.deepcopy(selection(self.current_generation))
            parent_2 = copy.deepcopy(selection(self.current_generation))

            child_1, child_2 = parent_1, parent_2

            can_crossover = (random.random() < self.crossover_probability)
            can_mutate = (random.random() < self.mutation_probability)

            if can_crossover:
                child_1.genes, child_2.genes = self.crossover_function(
                    parent_1.genes, parent_2.genes)

            if can_mutate:
                self.mutate_function(child_1.genes)
                self.mutate_function(child_2.genes)

            append_individual_in_new_population(child_1, new_individual_list, new_individual_gene_list, can_mutate,
                                                can_crossover)

            if len(new_individual_list) < self.population_size:
                append_individual_in_new_population(child_2, new_individual_list, new_individual_gene_list, can_mutate,
                                                    can_crossover)

        if self.elitism:
            # the new population will still retain the individual with the best fitness function from last generation.
            new_individual_list[0] = elite

        self.current_generation = new_individual_list

    def create_individual_gene(self):
        '''
        create individual's gene using the input parameter.
        self.param_number: number of parameters function as genes.
        self.param_range: range of parameter to select from.
        Here we choose continuous version of genetic algorithm.
        :param self:
        :return:
        '''
        return (np.random.random(self.parameter_number) * self.parameter_range).tolist()

    '''
    code to calculate fitness function for Genetic algorithm
    '''

    def calculate_population_fitness(self):
        """Calculate the fitness of every member of the given population using
        the supplied fitness_function.
        This function is the most time-consuming part for genetic algorithm.
        We only compute fitness function for mutated and cross-overed genes, which is recorded as 0 when we create new generation.
        """
        for individual in self.current_generation:
            # if fitness == 0. that means this is newly generated child. We have to compute its fitness function for new member.
            if individual.fitness == 0:
                # search if this genes is already in previous_genes list.
                # This approach is taken only when computation of cost function is extremely expensive
                if individual.genes not in self.previous_genes:
                    # fitness function is main computation part:  We run SUR algorithm and analyze P(t) results.
                    individual.fitness = self.fitness_function(
                        individual.genes, self.seed_data)

                    self.previous_genes.append(individual.genes)
                    self.previous_fitness.append(individual.fitness)
                else:
                    # already computed in the history. Record its value
                    index = self.previous_genes.index(individual.genes)
                    individual.fitness = self.previous_fitness[index]

    '''
    Code to immigrate the individual between different processes when we do multi-processing Genetic algorithm.
    '''

    def shuffle_genetic_data(self, immigrant_index_list):
        '''
        immigrant_index_list : index for individual in list to immigrate.
        self.immigrant_list: list of individual to migrate to other processes.
        used for immigration.
        :return:
        '''
        immigrant_fitness = np.array([member.fitness for member in self.immigrant_list])  # here member is chromosome
        immigrant_genes = np.array([member.genes for member in self.immigrant_list])  # gene for chromosome

        # random array
        # example : [1,2,3,4] -> [3,1, 4, 2]
        random_arr = []
        if rank == 0:
            random_arr = np.arange(num_proc)
            np.random.shuffle(random_arr)

        # shuffle the fitness function and gene according to random array (random_arr)
        immigrant_fitness = shuffle_data(immigrant_fitness, num_proc, random_arr)
        immigrant_genes = shuffle_data(immigrant_genes, num_proc, random_arr)

        # reconstruct chromosome
        for i in range(self.immigrant_num):
            self.immigrant_list[i].genes = immigrant_genes[i]
            self.immigrant_list[i].fitness = immigrant_fitness[i]

        # now we replace native chromosome with immigrant chromosome from other process
        for i in range(len(immigrant_index_list)):
            index = immigrant_index_list[i]

            # replace the individual in the current generation with the member from the immigration_list (from other process)
            member = self.immigrant_list[i]
            self.current_generation[index].fitness = member.fitness
            self.current_generation[index].genes = copy.deepcopy(member.genes).tolist()

    def immigrate_population(self):
        '''
        immigrate population between different processes for better genetic algorithm performance.
        :return:
        '''

        # clear immigrant list
        immigrant_index_list = []
        self.immigrant_list = []
        current_immigrant_num = 0

        if self.immigrant_num != 0:
            while current_immigrant_num < self.immigrant_num:
                immigrant_index = np.random.choice(self.population_size)
                if immigrant_index not in immigrant_index_list:
                    # we don't want to send same population to other process.
                    immigrant_index_list.append(immigrant_index)
                    current_immigrant_num = current_immigrant_num + 1

                    # immigrant is the chromosome
                    immigrant = self.current_generation[immigrant_index]
                    self.immigrant_list.append(copy.deepcopy(immigrant))

            # shuffle genetic data and replace native chromosome with immigrant chromosome.
            self.shuffle_genetic_data(immigrant_index_list)

    '''
    customized  mutation_function and crossover_function.
    Used to generate the child generation.
    '''

    def crossover_function(self,parent_1, parent_2):
        '''
        cross over function for continuous variable.
        :param parent_1: individual chromosome for crossover
        :param parent_2: individual chromosome for crossover
        :return:
        '''
        # rn in range [0,1]
        rn = np.random.random([self.parameter_number])

        parent_1_array = np.array(parent_1)
        parent_2_array = np.array(parent_2)

        child_1_array = parent_1_array - rn * (parent_1_array - parent_2_array)
        child_2_array = parent_2_array + rn * (parent_1_array - parent_2_array)

        child_1 = child_1_array.tolist()
        child_2 = child_2_array.tolist()

        return child_1, child_2

    def mutate_function(self, gene):
        '''
        mutate function for continuous variable. change one variable to random number
        :param gene: individual gene to do mutation
        :return:
        '''
        rn = random.random()
        mutate_index = random.randrange(self.parameter_number)
        gene[mutate_index] = rn * self.parameter_range





