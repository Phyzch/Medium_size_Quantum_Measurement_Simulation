from include.util import *
from pyeasyga import pyeasyga


# ---------------- Calculate fitness function for Genetic algorithm ----------------------------
def calculate_population_fitness(self):
    """Calculate the fitness of every member of the given population using
    the supplied fitness_function.
    This function is most time-consuming part for genetic algorithm.
     We only compute fitness function for mutated and crossovered genes, which is 0.
    """
    for individual in self.current_generation:
        # if fitness == 0. that means this is newly generated child or parents. We have to compute its fitness function.
        if individual.fitness == 0:
            # search if this genes is already in Biglist_genes list. This approach is taken only when computation of cost function is extremley expensive
            if (individual.genes not in self.Biglist_genes):
                # fitness function is main computation part:  We run SUR algorithm and analyze P(t) results.
                individual.fitness = self.fitness_function(
                    individual.genes, self.seed_data)
                self.Biglist_genes.append(individual.genes)
                self.Biglist_fitness.append(individual.fitness)
            else:
                # already computed in the history. Record its value
                index = self.Biglist_genes.index(individual.genes)
                individual.fitness = self.Biglist_fitness[index]

