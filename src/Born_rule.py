import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Fitness_function import simulate_full_system_energy_flow, Analyze_peak_and_peak_duration

from include.full_system_class.Full_system_class import full_system
from Feed_full_system_to_Genetic_algorithm import output_full_system_state_and_coupling_info
from include.util import Broadcast_data

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()

def set_detector_param():
    dof = 3
    frequency1 = np.array([1, 0.5, 0.25])
    nmax1 = [1, 2, 4]
    initial_d_state1 = [0, 0, 0]
    d1_energy_window = 1
    d1_state_coupling_energy_window = 10000   # no constraint now

    frequency2 = np.array([1, 0.5, 0.25])
    nmax2 = [1, 2, 4]
    initial_d_state2 = [0, 0, 0]
    d2_energy_window = 1
    d2_state_coupling_energy_window = 10000 # no constraint now.

    Detector_1_parameter = dof, frequency1, nmax1, initial_d_state1, d1_energy_window , d1_state_coupling_energy_window
    Detector_2_parameter = dof, frequency2, nmax2, initial_d_state2, d2_energy_window , d2_state_coupling_energy_window

    return Detector_1_parameter , Detector_2_parameter

def Analyze_Born_rule(file_path):
    # preview : True, see wave function figure. False, do batch simulation
    # before do batch simulation, we should preview simulation result and change time step, simulation time , coupling strength etc.
    preview = True
    save_preview_bool = True

    # use highest peak as criteria for localization.
    highest_peak_bool = False

    iteration_number = 1000

    # parameter_range is range for coupling strength we set in Hamiltonian.
    coupling_parameter_range = 0.01

    # ----------- parameter for photon ---------------------
    photon_energy = 1
    initial_photon_wavefunction = [np.sqrt(1) / np.sqrt(3), np.sqrt(2) / np.sqrt(3)]

    # -------- parameter for detector -----------
    Detector_1_parameter, Detector_2_parameter = set_detector_param()

    # - -------  specify input paramter for full system  (photon + detector)  ----------------
    # include state in full system (photon + detector) which satisfy energy condition  E - E_init <= energy window
    full_system_energy_window = 0

    # full system 's construct_full_system_Hamiltonian_part2() is called within fitness function in each cycle of genetic algorithm.
    full_system_instance = full_system(Detector_1_parameter, Detector_2_parameter, full_system_energy_window,
                                       photon_energy, initial_photon_wavefunction)

    full_system_instance.construct_full_system_Hamiltonian_part1()

    # print information about structure of system
    output_full_system_state_and_coupling_info(full_system_instance)

    parameter_number = full_system_instance.output_offdiagonal_parameter_number()

    left_localization_num = 0
    right_localization_num = 0

    parameter_list = []
    max_energy_change_list = []
    localization_side_list = []

    iteration_number_per_core = int (iteration_number / num_proc)
    iteration_number = iteration_number_per_core * num_proc

    if(preview):
        plot_trail_simulation_result(full_system_instance, coupling_parameter_range, parameter_number, file_path, save_preview_bool)
    else:
        for i in range(iteration_number_per_core):
            # randomly generate parameter according to coupling_parameter_range:
            coupling_param = np.random.normal(0, coupling_parameter_range, parameter_number).tolist()
            assert(type(coupling_param) == list )

            photon_energy_list, d1_energy_list_change, d2_energy_list_change, time_list = simulate_full_system_energy_flow(full_system_instance, coupling_param)

            _, max_energy_change, _, localization_bool = Analyze_peak_and_peak_duration(
                d1_energy_list_change, d2_energy_list_change, time_list , highest_peak_bool= highest_peak_bool)

            if(localization_bool == 1):
                left_localization_num = left_localization_num + 1

            if (localization_bool == 2 ):
                right_localization_num = right_localization_num + 1

            parameter_list.append(coupling_param)
            max_energy_change_list.append(max_energy_change)
            localization_side_list.append(localization_bool)

        # Broadcast data to all process.
        parameter_list = Broadcast_data(parameter_list , num_proc )
        max_energy_change_list = Broadcast_data(max_energy_change_list , num_proc)
        localization_side_list = Broadcast_data(localization_side_list , num_proc )

        Analyze_Localization_prob( max_energy_change_list, localization_side_list, parameter_list, initial_photon_wavefunction, iteration_number_per_core, iteration_number, file_path)


def plot_trail_simulation_result(full_system_instance, coupling_parameter_range, parameter_number, file_path, save_preview_bool ):
    if rank == 0:
        coupling_param = np.random.normal(0, coupling_parameter_range, parameter_number).tolist()

        photon_energy_list, d1_energy_list_change, d2_energy_list_change, time_list = simulate_full_system_energy_flow(
            full_system_instance, coupling_param)

        tot_energy = photon_energy_list + d1_energy_list_change + d2_energy_list_change

        #configure figure
        fig = plt.figure(figsize=(20, 10))
        spec = gridspec.GridSpec(nrows=1, ncols=1, figure=fig)
        spec.update(hspace=0.5, wspace=0.3)
        ax = fig.add_subplot(spec[0, 0])

        ax.plot(time_list, photon_energy_list , color = 'orange' , label = '$E_{p}$')
        ax.plot(time_list , d1_energy_list_change , color = 'blue' , label = '$E_{l}$')
        ax.plot(time_list, d2_energy_list_change , color = 'green' , label = '$E_{r}$')
        ax.plot(time_list, tot_energy , color = 'red' , label = '$E_{tot}$')

        ax.legend(loc = 'best')
        ax.set_xlabel('time')
        ax.set_ylabel('E')
        ax.set_title('energy exchange')

        if save_preview_bool:
            fig_name = "quantum measurement process preview.svg"
            fig_path = os.path.join(file_path, fig_name)

            fig.savefig(fig_path)

        plt.show()


def Analyze_Localization_prob( max_energy_change_list, localization_side_list, parameter_list, psi0, iteration_number_per_core, iteration_number, file_path):
    if (rank == 0):
        sample_num = len(localization_side_list)

        left_side_sample_num = 0
        right_side_sample_num = 0
        left_localization_energy_list = []
        right_localization_energy_list = [ ]
        for i in range(sample_num):
            if localization_side_list[i] == 1:
                left_side_sample_num = left_side_sample_num + 1
                left_localization_energy_list.append(max_energy_change_list[i])
            if localization_side_list[i] == 2:
                right_side_sample_num = right_side_sample_num + 1
                right_localization_energy_list.append(max_energy_change_list[i])

        left_side_prob = left_side_sample_num / sample_num
        right_side_prob = right_side_sample_num / sample_num

        left_max_energy_change_list = [max_energy_change_list[i] for i in range(len(max_energy_change_list)) if localization_side_list[i] == 1]
        right_max_energy_change_list = [max_energy_change_list[i] for i in range(len(max_energy_change_list)) if
                                        localization_side_list[i] == 2]

        Born_prob = np.power(psi0 , 2)


        # compute localization probability after we sift result with maximum energy criteria.
        criteria_list, tot_sample_num_satisfy_criteria_list, left_side_prob_satisfy_criteria_list, right_side_prob_satisfy_criteria_list \
            = compute_localization_prob_satisfy_criteria(left_localization_energy_list , right_localization_energy_list)

        # print information to screen
        print_info(iteration_number , iteration_number_per_core , psi0, Born_prob, left_side_prob, right_side_prob , sample_num ,
                   max_energy_change_list, left_max_energy_change_list , right_max_energy_change_list)

        # output localization info
        output_localization_info(criteria_list, tot_sample_num_satisfy_criteria_list , left_side_prob_satisfy_criteria_list , right_side_prob_satisfy_criteria_list , file_path)

        # output parameter
        output_param(parameter_list, file_path)

        # output localization information and energy exchange for localization to left
        left_localize_param_file_name = "localization_parameter_list_left.txt"
        output_localization_param_and_max_energy_change(psi0, max_energy_change_list, localization_side_list, parameter_list, left_localize_param_file_name,
                                                        file_path, localization_symbol= 1)

        # output localization information and energy exchange for localization to right
        right_localize_param_file_name = "localization parameter list right.txt"
        output_localization_param_and_max_energy_change(psi0, max_energy_change_list, localization_side_list, parameter_list, right_localize_param_file_name,
                                                        file_path , localization_symbol=  2)

def compute_localization_prob_satisfy_criteria(left_localization_energy_list , right_localization_energy_list):
    criteria_list = [0, 0.6, 0.7, 0.8, 0.9]
    tot_sample_num_satisfy_criteria_list = []
    left_side_prob_satisfy_criteria_list = []
    right_side_prob_satisfy_criteria_list = []

    for criteria in criteria_list:
        left_side_sample_num = [i for i in left_localization_energy_list if i >= criteria]
        left_side_num = len(left_side_sample_num)
        right_side_sample_num = [i for i in right_localization_energy_list if i >= criteria]
        right_side_num = len(right_side_sample_num)

        tot_sample_num_satisfy_criteria = left_side_num + right_side_num

        if tot_sample_num_satisfy_criteria != 0:
            left_side_prob_satisfy_criteria = len(left_side_sample_num) / tot_sample_num_satisfy_criteria
            right_side_prob_satisfy_criteria = len(right_side_sample_num) / tot_sample_num_satisfy_criteria
        else:
            left_side_prob_satisfy_criteria = 0
            right_side_prob_satisfy_criteria = 0

        tot_sample_num_satisfy_criteria_list.append(tot_sample_num_satisfy_criteria)
        left_side_prob_satisfy_criteria_list.append(left_side_prob_satisfy_criteria)
        right_side_prob_satisfy_criteria_list.append(right_side_prob_satisfy_criteria)

    return criteria_list , tot_sample_num_satisfy_criteria_list , left_side_prob_satisfy_criteria_list , right_side_prob_satisfy_criteria_list

def print_info(*args):

    iteration_number , iteration_number_per_core , psi0, Born_prob, left_side_prob, right_side_prob , sample_num ,\
    max_energy_change_list, left_max_energy_change_list , right_max_energy_change_list = args

    print('number of core:  ')
    print(num_proc)

    print('total iteration number: ')
    print(iteration_number)

    print('iteartion number per core: ')
    print(iteration_number_per_core)

    print("wave function: " + str(psi0))
    print("Born rule prob (theoretical):  " + str(Born_prob))
    print("Left side prob:  " + str(left_side_prob))
    print("Right side prob: " + str(right_side_prob))
    print('total sample number ' + str(sample_num))

    print('Mean localization energy:  ')
    print(np.mean(max_energy_change_list))

    print('Mean localization energy left:  ')
    print(np.mean(left_max_energy_change_list))

    print("Mean localization energy right:  ")
    print(np.mean(right_max_energy_change_list))

def output_localization_info(*args):

    criteria_list, tot_sample_num_satisfy_criteria_list , left_side_prob_satisfy_criteria_list , right_side_prob_satisfy_criteria_list , file_path = args
    filename = 'Localization_analyze_result.txt'
    filename = os.path.join(file_path, filename)

    with open(filename, 'w') as f:
        f.write('Critera   Total_number_sample  left_prob   right_prob')
        f.write('\n')

        for i in range(len(criteria_list)):
            criteria = criteria_list[i]
            total_num_after_sift = tot_sample_num_satisfy_criteria_list[i]
            left_side_prob_satisfy_criteria = left_side_prob_satisfy_criteria_list[i]
            right_side_prob_satisfy_criteria = right_side_prob_satisfy_criteria_list[i]

            f.write(str(criteria) + "  " + str(total_num_after_sift) + "  " + str(
                left_side_prob_satisfy_criteria) + "  " + str(right_side_prob_satisfy_criteria) + "\n")

def output_param(parameter_list, file_path):
    # write parameter:
    filename1 = "parameter_for_simulation.txt"
    filename1 = os.path.join(file_path, filename1)

    with open(filename1, 'w') as f1:
        f1.write('parameter set : \n')
        parameter_list_len = len(parameter_list)

        for i in range(parameter_list_len):
            parameter_set = parameter_list[i]
            for number in parameter_set:
                f1.write(str(number) + " ")
            f1.write('\n')

def output_localization_param_and_max_energy_change( psi0, max_energy_change_list , localization_side_list , parameter_list , file_name , file_path , localization_symbol):
    '''

    :param psi0:
    :param max_energy_change_list:
    :param localization_side_list:
    :param parameter_list:
    :param file_name:
    :param file_path:
    :param localization_symbol: localization symbol == 1: localize to left.  localization symbol == 2 : localize to right.
    :return:
    '''

    file_name = os.path.join(file_path, file_name)

    with open(file_name, "w") as f2:
        for j in psi0:
            f2.write(str(j) + " ")
        f2.write('\n')

        parameter_list_len = len(max_energy_change_list)
        for i in range(parameter_list_len):
            if (localization_side_list[i] ==  localization_symbol ):
                parameter = parameter_list[i]
                for j in parameter:
                    f2.write(str(j) + ' ')
                f2.write('\n')
                f2.write(str(max_energy_change_list[i]) + "\n")

