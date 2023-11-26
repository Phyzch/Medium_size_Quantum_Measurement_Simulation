from pyeasyga import pyeasyga
from include.util import *



# Need to specify fitness function for Genetic algorithm module
# ga.fitness_function = customized_fitness_function.  Fitnessfunction should have form : function(genes, seed_param) . Here seed_param is parameters we feed in.
# You have to define your own create_individual and crossover function.
# This is continuous genetic algorithm, which means paramter to be optimized is continuous. See: Section 3 of Practical Genetic algorithms (Second Edition) -- Randy L. Haupt, Sue Ellen Haupt.
# class below inherit from pyeasyga class : see https://pypi.org/project/pyeasyga/


class Extend_Genetic_algorithm(pyeasyga.GeneticAlgorithm):

    # import method.
    from create_generation import create_first_generation, create_next_generation, create_initial_population, \
        create_new_population, create_individual_gene
    from immigrate import immigrate_population, shuffle_genetic_data
    from mutate_crossover import  mutate_function1, crossover_function1
    from fitness_func import calculate_population_fitness


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
        param_range , _ ,  param_number , _ = self.seed_data
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







