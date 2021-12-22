import sys

add_sys_path_list = ["/home/phyzch/PycharmProjects/Medium_size_Quantum_Measurement_simulation/" , "/home/phyzch/PycharmProjects/Medium_size_Quantum_Measurement_simulation/src/" ,
                     "/home/phyzch/PycharmProjects/Medium_size_Quantum_Measurement_simulation/include/detector_class/",
                     "/home/phyzch/PycharmProjects/Medium_size_Quantum_Measurement_simulation/include/full_system_class/",
                     "/home/phyzch/PycharmProjects/Medium_size_Quantum_Measurement_simulation/include/genetic_algorithm_class/"]

sys.path = sys.path + add_sys_path_list

from Feed_full_system_to_Genetic_algorithm import Implement_genetic_algorithm
from Born_rule import Analyze_Born_rule
import matplotlib

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()

def main():
    '''

    :return:
    '''
    matplotlib.rcParams.update({'font.size': 20})

    # print_sys_path_info()

    # file_path = "/home/phyzch/PycharmProjects/Medium_size_Quantum_Measurement_simulation/result/Genetic_algorithm/3_mode/"
    # Implement_genetic_algorithm(file_path)


    Analyze_Born_rule_file_path = "/home/phyzch/PycharmProjects/Medium_size_Quantum_Measurement_simulation/result/Born_rule/try/"
    Analyze_Born_rule(Analyze_Born_rule_file_path)

def print_sys_path_info():
    print(sys.path)

main()