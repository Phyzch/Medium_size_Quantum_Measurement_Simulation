from include.util import *
from pyeasyga import pyeasyga


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

    def append_individual_in_new_population(individual, new_population, new_population_gene, can_mutate, can_crossover):
        # we should not have  same genes in one generation, this may lead us go to local minimum
        if (not (individual.genes in new_population_gene)):
            if (can_mutate or can_crossover):
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

        append_individual_in_new_population(child_1, new_population, new_population_gene, can_mutate, can_crossover)

        if len(new_population) < self.population_size:
            append_individual_in_new_population(child_2, new_population, new_population_gene, can_mutate, can_crossover)

    if self.elitism:
        new_population[0] = elite

    self.current_generation = new_population


def create_individual_gene(self):
    return [np.random.random(self.param_number) * self.param_range]

