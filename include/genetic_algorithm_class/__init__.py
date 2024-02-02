from pyeasyga import pyeasyga
from timeit import default_timer as timer
from include.util import rank

'''
Need to specify fitness function for Genetic algorithm module
ga.fitness_function = customized_fitness_function.  Fitnessfunction should have form : function(genes, seed_param) . Here seed_param is parameters we feed in.
You have to define your own create_individual and crossover function.
This is continuous genetic algorithm, which means paramter to be optimized is continuous. See: Section 3 of Practical Genetic algorithms (Second Edition) -- Randy L. Haupt, Sue Ellen Haupt.
class below inherit from pyeasyga class : see https://pypi.org/project/pyeasyga/
'''

class ExtendGeneticAlgorithm(pyeasyga.GeneticAlgorithm):
    '''
    Extended Genetic algorithm class.
    This class is defined as inheritance of pyeasyga.GeneticAlgorithm class (https://pyeasyga.readthedocs.io/en/latest/)

    '''
    # import method.
    from create_generation import create_first_generation, create_next_generation, create_initial_population, \
        create_new_population, create_individual_gene
    from immigrate import immigrate_population, shuffle_genetic_data
    from mutate_and_crossover import  mutate_function, crossover_function
    from fitness_func import calculate_population_fitness


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





