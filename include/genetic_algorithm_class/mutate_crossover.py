from include.util import *

# -------------- customized create_individual function , mutation_function and crossover_function.  -----------------
def crossover_function1(self, parent_1, parent_2):
    '''
    cross over function for continuous variable.
    :param parent_1:
    :param parent_2:
    :return:
    '''
    # rn in range [0,1]
    rn = np.random.random([self.param_number])
    parent_1_array = np.array(parent_1)
    parent_2_array = np.array(parent_2)
    child_1_array = parent_1_array - rn * (parent_1_array - parent_2_array)
    child_2_array = parent_2_array + rn * (parent_1_array - parent_2_array)

    child_1 = child_1_array.tolist()
    child_2 = child_2_array.tolist()

    return child_1, child_2


def mutate_function1(self, gene):
    '''
    mutate function for continuous variable. change one variable to random number
    :param gene: gene to do mutation
    :return:
    '''
    rn = random.random()
    mutate_index = random.randrange(self.param_number)
    gene[mutate_index] = rn * self.param_range

