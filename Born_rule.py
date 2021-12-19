import Shared_data
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib
from Fitness_function import Evolve_full_system_and_return_energy_change, Analyze_peak_and_peak_duration

from Full_system_class import full_system

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()


def Analyze_Born_rule(file_path):
    iteration_number = 100

    coupling_strength = 0.1

    photon_energy = 1

    dof = 3

    frequency1 = [1, 0.5 , 0.25 ]
    frequency2 = [1, 0.5 , 0.25]

    frequency1 = np.array(frequency1)
    frequency2 = np.array(frequency2)

    nmax1 = [1, 2, 4 ]
    nmax2 = [1, 2, 4]

    initial_state1 = [0, 0, 0]
    initial_state2 = [0, 0, 0]

    energy_window1 = 1
    energy_window2 = 1

    full_system_energy_window = 0

    Detector_1_parameter = dof, frequency1, nmax1, initial_state1, energy_window1
    Detector_2_parameter = dof, frequency2, nmax2, initial_state2, energy_window2

    Initial_Wavefunction = [np.sqrt(1) / np.sqrt(4), np.sqrt(3) / np.sqrt(4)]

    # Other part of code for full system is called within fitness function in each cycle of genetic algorithm.
    full_system_instance = full_system(Detector_1_parameter, Detector_2_parameter, full_system_energy_window,
                                       photon_energy, Initial_Wavefunction)
    full_system_instance.construct_full_system_Hamiltonian_part1()

    # print information about structure of system
    if(rank == 0):
        full_system_instance.print_state_mode()
        full_system_instance.detector1.output_detector_state_coupling()
        full_system_instance.output_off_diagonal_coupling_mode_info()

        print( "parameter number for detector1: "  + str(full_system_instance.detector1.offdiag_coupling_num) )
        print( "parameter number for detector2: " + str(full_system_instance.detector2.offdiag_coupling_num) )
        print( "paramter number for coupling betweeen detector and system:  " + str(full_system_instance.offdiagonal_parameter_number -
                                                                                    full_system_instance.detector1.offdiag_coupling_num -
                                                                                    full_system_instance.detector2.offdiag_coupling_num))

    parameter_number = full_system_instance.output_offdiagonal_parameter_number()

    Left_localization_number = 0
    Right_localization_number = 0

    parameter_list = []
    Max_energy_change_list = []
    Localization_side_list = []

    iteration_number_per_core = int (iteration_number / num_proc)
    iteration_number = iteration_number_per_core * num_proc

    for i in range(iteration_number_per_core):
        # randomly generate parameter according to coupling strength:
        Coupling_param = np.random.normal(0, coupling_strength, parameter_number)

        photon_energy_list, d1_energy_list_change, d2_energy_list_change, Time_list = Evolve_full_system_and_return_energy_change(full_system_instance , Coupling_param)

        First_peak_Time_duration, max_energy_change, Localization_duration_ratio, localization_bool = Analyze_peak_and_peak_duration(
            d1_energy_list_change, d2_energy_list_change, Time_list)

        if(localization_bool == 1):
            Left_localization_number = Left_localization_number + 1

        if(localization_bool == 2):
            Right_localization_number = Right_localization_number + 1

        parameter_list.append(Coupling_param)
        Max_energy_change_list.append(max_energy_change)
        Localization_side_list.append(localization_bool)

    # shape:  [iteration_per_core]
    Max_energy_change_list = np.array(Max_energy_change_list)
    Localization_side_list = np.array(Localization_side_list)
    # shape: [iteration_per_core , parameter_number ]
    parameter_list = np.array(parameter_list)

    recv_max_energy_change_list = []
    recv_localization_list = []
    recv_parameter_list = []

    if(rank == 0):
        recv_max_energy_change_list = np.empty([num_proc, iteration_number_per_core] , dtype = np.float64)
        recv_localization_list = np.empty([num_proc, iteration_number_per_core], dtype = np.int64)
        recv_parameter_list = np.empty([num_proc, iteration_number_per_core, parameter_number] , dtype = np.float64)

    comm.Gather(Max_energy_change_list , recv_max_energy_change_list , 0)
    comm.Gather(Localization_side_list, recv_localization_list , 0)
    comm.Gather(parameter_list , recv_parameter_list, 0)

    if(rank == 0):
        Max_energy_change_list = np.reshape(recv_max_energy_change_list,
                                            (recv_max_energy_change_list.shape[0] * recv_max_energy_change_list.shape[1]))
        Localization_side_list = np.reshape(recv_localization_list,
                                            (recv_localization_list.shape[0] * recv_localization_list.shape[1]))
        parameter_list = np.reshape(recv_parameter_list, (
        recv_parameter_list.shape[0] * recv_parameter_list.shape[1], recv_parameter_list.shape[2]))

    Analyze_Localization_prob(iteration_number_per_core, Max_energy_change_list, Localization_side_list, parameter_list, Initial_Wavefunction, iteration_number, file_path)




def Analyze_Localization_prob(iteration_number_per_core,Max_energy_change_list , Localization_side_list, parameter_list , psi0 , iteration_number, file_path):
    if (rank == 0):
        print('number of core:  ')
        print(num_proc)

        print('total iteration number: ')
        print(iteration_number)

        print('iteartion number per core: ')
        print(iteration_number_per_core)

        List_len = len(Localization_side_list)

        Left_side_time = 0
        Right_side_time = 0
        Left_localization_energy_list = []
        Right_localization_energy_list = [ ]
        for i in range(List_len):
            if Localization_side_list[i] == 1:
                Left_side_time = Left_side_time + 1
                Left_localization_energy_list.append(Max_energy_change_list[i])
            if Localization_side_list[i] == 2:
                Right_side_time = Right_side_time + 1
                Right_localization_energy_list.append(Max_energy_change_list[i])

        Left_side_prob = Left_side_time / List_len
        Right_side_prob = Right_side_time / List_len
        print("wave function: " + str(psi0))
        Born_prob = np.power(psi0 , 2)
        print("Born rule prob (theoretical):  " + str(Born_prob))
        print("Left side prob:  " + str(Left_side_prob))
        print("Right side prob: " + str(Right_side_prob))
        print('total sample number ' + str(List_len))

        Criteria_list = [ 0 , 0.6, 0.7, 0.8, 0.9]
        Total_num_after_sift_list = []
        Left_side_prob_after_sift_list = []
        Right_side_prob_after_sift_list = []
        for Criteria in Criteria_list:
            Left_localization_energy_list_after_sift = [i for i in Left_localization_energy_list if i >= Criteria]
            Right_localization_energy_list_after_sift = [i for i in Right_localization_energy_list if i >= Criteria]
            Total_sample_num_after_sift = len(Left_localization_energy_list_after_sift) + len(
                Right_localization_energy_list_after_sift)
            if(Total_sample_num_after_sift!=0):
                Left_side_prob_after_sift = len(Left_localization_energy_list_after_sift) / Total_sample_num_after_sift
                Right_side_prob_after_sift = len(Right_localization_energy_list_after_sift) / Total_sample_num_after_sift
            else:
                Left_side_prob_after_sift = 0
                Right_side_prob_after_sift = 0

            Total_num_after_sift_list.append(Total_sample_num_after_sift)
            Left_side_prob_after_sift_list.append(Left_side_prob_after_sift)
            Right_side_prob_after_sift_list.append(Right_side_prob_after_sift)


        # print('Localization criteria:  ' + str(Criteria) + "  .  total sample number:  " + str(
        #     Total_sample_num_after_sift))
        # print('Left side prob:  ' + str(Left_side_prob_after_sift) + "      Right side prob:   " + str(
        #     Right_side_prob_after_sift))

        filename = 'Localization_analyze_result.txt'
        filename = os.path.join(file_path, filename)

        with open(filename, 'w') as f:
            f.write('Critera   Total_number_sample  left_prob   right_prob')
            f.write('\n')

            for i in range(len(Criteria_list)):
                Criteria = Criteria_list[i]
                Total_num_after_sift =  Total_num_after_sift_list[i]
                Left_side_prob_after_sift = Left_side_prob_after_sift_list[i]
                Right_side_prob_after_sift = Right_side_prob_after_sift_list[i]

                f.write( str(Criteria) + "  " +  str(Total_num_after_sift) + "  " + str(Left_side_prob_after_sift) + "  " + str(Right_side_prob_after_sift) + "\n" )

        # write parameter:
        filename1 = "parameter_for_simulation.txt"
        filename1 = os.path.join(file_path, filename1)

        with open(filename1 , 'w') as f1:
            f1.write('parameter set : \n')
            parameter_list_len = len(parameter_list)

            for i in range(parameter_list_len):
                parameter_set = parameter_list[i]
                for number in parameter_set:
                    f1.write(str(number) + " ")
                f1.write('\n')

        filename2 = "localization_parameter_list_left.txt"
        filename2 = os.path.join(file_path, filename2)

        Left_Max_energy_change_list = [ Max_energy_change_list[i] for i in range(len(Max_energy_change_list)) if Localization_side_list[i] == 1  ]

        with open(filename2 , "w") as f2:
            for j in psi0:
                f2.write(str(j) + " ")
            f2.write('\n')

            parameter_list_len  = len(Max_energy_change_list)
            for i in range(parameter_list_len):
                if(Localization_side_list[i] == 1):
                    parameter = parameter_list[i]
                    for j in parameter:
                        f2.write(str(j) + ' ')
                    f2.write('\n')
                    f2.write(str(Max_energy_change_list[i]) + "\n")

        Right_Max_energy_change_list = [Max_energy_change_list[i] for i in range(len(Max_energy_change_list)) if
                                        Localization_side_list[i] == 2]

        filename3 = "localization_parameter_list_right.txt"
        filename3 = os.path.join(file_path,filename3)

        with open(filename3 ,"w" ) as f3:
            for j in psi0:
                f3.write(str(j) + " ")

            f3.write('\n')

            parameter_len = len(Max_energy_change_list)
            for i in range(parameter_len):
                if Localization_side_list[i] == 2:

                    parameter = parameter_list[i]
                    for j in parameter:
                        f3.write(str(j) + ' ')
                    f3.write('\n')
                    f3.write(str(Max_energy_change_list[i]) + "\n")

        print('Mean localization energy:  ')
        print(np.mean(Max_energy_change_list))

        print('Mean localization energy left:  ' )
        print(np.mean(Left_Max_energy_change_list))

        print("Mean localization energy right:  ")
        print(np.mean(Right_Max_energy_change_list))






