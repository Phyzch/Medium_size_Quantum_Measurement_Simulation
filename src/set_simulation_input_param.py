import os
import numpy as np
from include.util import rank, num_proc

def set_detector_param():
    '''

    :return: detector1_paramter, detector2_parameter
    '''
    # number of degrees of freedom
    dof = 4
    # -------------- parameter for d1 -------------------------
    frequency_detector1 = np.array([1, 0.5, 0.25, 0.125])     # frequency of detector's vibration mode

    max_qn_detector1 = [1, 2, 4, 8]     # maximum quantum number when constructing basis set.

    initial_detector1_state_qn = [0, 0, 0, 0]     # initial detector states' quantum number

    detector1_energy_window = 1     # energy window for detector to include state around init state.

    # state couple to each other in detector should satisfy \Delta E <= state_coupling_energy_window.
    # set state_coupling_energy_window == 0 means only resonant state can couple to each other.
    detector1_state_coupling_energy_window = 0

    # -------------- parameter for d2 ----------------
    frequency_detector2 = np.array([1, 0.5, 0.25, 0.125])
    max_qn_detector2 = [1, 2, 4, 8]
    initial_detector2_state_qn = [0, 0, 0, 0]
    detector2_energy_window = 1
    detector2_state_coupling_energy_window = 0

    detector1_parameter = (dof, frequency_detector1, max_qn_detector1, initial_detector1_state_qn,
                            detector1_energy_window, detector1_state_coupling_energy_window)
    detector2_parameter = (dof, frequency_detector2, max_qn_detector2, initial_detector2_state_qn,
                            detector2_energy_window, detector2_state_coupling_energy_window)

    return detector1_parameter, detector2_parameter


def prepare_genetic_algorithm_info_file(file_path, population_size_each_process, population_size_over_all_process):
    '''

    :param file_path: path to the info file
    :param population_size_each_process: size of genetic algorithm population in each process
    :param population_size_over_all_process: size of genetic algorithm population across all processes.
    :return:
    '''
    # info file for Genetic algorithm.
    f2 = 0

    if rank == 0:
        filename = 'info.txt'
        filename = os.path.join(file_path, filename)
        f2 = open(filename, 'w')
        f2.write('population size:  ' + str(population_size_each_process) + '\n')
        f2.write('population size in all processes:  ' + str(population_size_over_all_process) + '\n')
        f2.write('number of processes:  ' + str(num_proc) + '\n')

    return f2
