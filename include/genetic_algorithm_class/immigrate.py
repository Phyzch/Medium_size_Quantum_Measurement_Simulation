from include.util import *
from pyeasyga import pyeasyga

# ----------------------- immigration module ------------------------------------
def shuffle_genetic_data(self , immigrant_index_list ):
    '''
    immigrant_index_list : index for individual in list to immigrate.
    used for immigration.
    :return:
    '''
    immigrant_fitness = np.array([member.fitness for member in self.immigrant_list])
    immigrant_genes = np.array([member.genes for member in self.immigrant_list])

    arr_random = []
    if(rank == 0):
        arr_random = np.arange(num_proc)
        np.random.shuffle(arr_random)

    immigrant_fitness = shuffle_data(immigrant_fitness , num_proc, arr_random)
    immigrant_genes = shuffle_data(immigrant_genes , num_proc, arr_random)

    # reconstruct chromosome
    for i in range(self.immigrant_num):
        self.immigrant_list[i].genes = immigrant_genes[i]
        self.immigrant_list[i].fitness = immigrant_fitness[i]

    # now we replace native chromosome with immigrant chromosome from other process
    for i in range(len(immigrant_index_list)):
        index = immigrant_index_list[i]

        member = self.immigrant_list[i]
        self.current_generation[index].fitness = member.fitness
        self.current_generation[index].genes = copy.deepcopy(member.genes).tolist()

def immigrate_population(self):
    '''
    immigrate population between different process for better genetic algorithm performance.
    :return:
    '''

    # clear immigrant list
    immigrant_index_list = []
    self.immigrant_list = []
    current_immigrant_num = 0

    if(self.immigrant_num != 0):
        while(current_immigrant_num < self.immigrant_num):
            immigrant_index  = np.random.choice(self.population_size)
            # we don't want to send same population to other process.
            if immigrant_index not in immigrant_index_list:
                immigrant_index_list.append(immigrant_index)
                current_immigrant_num = current_immigrant_num + 1

                immigrant = self.current_generation[immigrant_index]
                self.immigrant_list.append(  copy.deepcopy(immigrant) )

        # shuffle genetic data and replace native chromosome with immigrant chromosome.
        self.shuffle_genetic_data(immigrant_index_list)
# --------------- immigration module ------------------------------------
