from include.util import *
from pyeasyga import pyeasyga


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

    def append_individual_in_new_population(individual, new_individual_list, new_individual_gene_list, mutate_bool, crossover_bool):
        '''

        :param individual: individual in population for genetic algorithm
        :type: individual: chromosome
        :param new_individual_list: a list of new individual chromosome
        :param new_individual_gene_list: The gene of new individuals
        :param mutate_bool: if True, mutate the genes.
        :param crossover_bool: if True, do crossover to genes
        :return:
        '''
        # we should not have the same genes in one generation, this may lead us go to local minimum
        if not (individual.genes in new_individual_gene_list):
            if mutate_bool or crossover_bool:
                # only re-compute fitness function if there is a need for mutation or crossover.
                # in the program, we will only recompute fitness function when it is 0.
                individual.fitness = 0

            new_individual_list.append(individual)
            new_individual_gene_list.append(individual.genes)

    new_individual_list = []   # list of individual chromosome for new generation.
    new_individual_gene_list = [] # list of individual chromosome's gene for new generation.
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
                parent_1.genes, parent_2.genes, self.param_number)

        if can_mutate:
            self.mutate_function(child_1.genes, self.param_number, self.param_range)
            self.mutate_function(child_2.genes, self.param_number, self.param_range)

        append_individual_in_new_population(child_1, new_individual_list, new_individual_gene_list, can_mutate, can_crossover)

        if len(new_individual_list) < self.population_size:
            append_individual_in_new_population(child_2, new_individual_list, new_individual_gene_list, can_mutate, can_crossover)

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
    return (np.random.random(self.param_number) * self.param_range).tolist()

