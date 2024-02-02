from pyeasyga import pyeasyga

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
            if  individual.genes not in self.previous_genes:
                # fitness function is main computation part:  We run SUR algorithm and analyze P(t) results.
                individual.fitness = self.fitness_function_for_individual_full_system_instance(
                    individual.genes, self.seed_data)
                self.previous_genes.append(individual.genes)
                self.previous_fitness.append(individual.fitness)
            else:
                # already computed in the history. Record its value
                index = self.previous_genes.index(individual.genes)
                individual.fitness = self.previous_fitness[index]

