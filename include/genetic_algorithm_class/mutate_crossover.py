from include.util import *

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

