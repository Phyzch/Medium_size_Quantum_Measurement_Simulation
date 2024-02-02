# in general, we should avoid using syntax like from ***(module) import *
# the best way to import module is actually import module, see https://python-docs.readthedocs.io/en/latest/writing/structure.html
import sys
import os
import feed_full_system_to_Genetic_algorithm
import matplotlib

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()

home_directory = "/home/phyzch/PycharmProjects/Medium_size_Quantum_Measurement_simulation/"
# add search directory path to sys.path.
add_sys_path_list = [home_directory , os.path.join(home_directory , "src" )  ,
                     os.path.join(home_directory , "include/detector/"),
                     os.path.join(home_directory , "include/fullsystem/") ,
                     os.path.join(home_directory , "include/genetic_algorithm_class/")]

sys.path = sys.path + add_sys_path_list

def main():
    '''

    :return:
    '''
    matplotlib.rcParams.update({'font.size': 20})

    file_path = "/home/phyzch/PycharmProjects/Medium_size_Quantum_Measurement_simulation/result/Genetic_algorithm/try/"
    feed_full_system_to_Genetic_algorithm.implement_genetic_algorithm(file_path)


main()