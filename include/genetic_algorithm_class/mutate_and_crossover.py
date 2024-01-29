from include.util import *

# -------------- customized create_individual function , mutation_function and crossover_function.  -----------------
def crossover_function(parent_1, parent_2, parameter_number):
    '''
    cross over function for continuous variable.
    :param parent_1: individual chromosome for crossover
    :param parent_2: individual chromosome for crossover
    :param parameter_number: number of parameters for crossover
    :return:
    '''
    # rn in range [0,1]
    rn = np.random.random([parameter_number])

    parent_1_array = np.array(parent_1)
    parent_2_array = np.array(parent_2)

    child_1_array = parent_1_array - rn * (parent_1_array - parent_2_array)
    child_2_array = parent_2_array + rn * (parent_1_array - parent_2_array)

    child_1 = child_1_array.tolist()
    child_2 = child_2_array.tolist()

    return child_1, child_2


def mutate_function(gene, parameter_number, parameter_range):
    '''
    mutate function for continuous variable. change one variable to random number
    :param gene: individual gene to do mutation
    :param parameter_number: number of parameters for mutation.
    :param parameter_range: range of parameter to perform mutation.
    :return:
    '''
    rn = random.random()
    mutate_index = random.randrange(parameter_number)
    gene[mutate_index] = rn * parameter_range

