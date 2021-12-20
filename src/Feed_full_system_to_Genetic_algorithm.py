import numpy as np
import matplotlib.pyplot as plt

from include.Full_system_class import full_system

from include.Genetic_algorithm_class import Extend_Genetic_algorithm
from Fitness_function import fitness_function , Evolve_full_system_and_return_energy_change, Analyze_peak_and_peak_duration
from Fitness_function import Convert_bit_to_parameter, fit_func1
from include.util import Broadcast_data
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()
import os


def plot_simulation_result( *args ):
    Time_list , photon_energy_list, d1_energy_list_change, d2_energy_list_change , file_path = args
    # plot simulation result
    fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.plot(Time_list, d1_energy_list_change, label='left photon localization')
    ax1.plot(Time_list, d2_energy_list_change, label='right photon localization')
    ax1.plot(Time_list, photon_energy_list, label='photon energy')

    ax1.legend(loc='best')

    fig_name = "best_simulation_result.png"
    fig_name = os.path.join(file_path, fig_name)

    fig1.savefig(fig_name)

def save_simulation_result(*args):
    best_param , best_fitness, parameter_number, coupling_strength, Time_list, \
    First_peak_Time_duration_best_param , max_energy_change_best_param , Localization_duration_ratio_best_param ,\
    d1_energy_list_change , d2_energy_list_change , photon_energy_list , file_path = args

    # save best simulation result found
    filename1 = 'best_parameter_and_fitness_function.txt'
    filename1 = os.path.join(file_path, filename1)
    with open(filename1, 'w') as f1:

        for num in best_param:
            f1.write(str(num) + " , ")
        f1.write('\n')

        # compute geometric mean and write to output file.
        parameter_set_geometric_mean = 1

        for parameter in best_param:
            parameter_set_geometric_mean = parameter_set_geometric_mean * np.power(np.abs(parameter),
                                                                                   1 / parameter_number)

        parameter_set_geometric_mean_ratio = parameter_set_geometric_mean / coupling_strength

        f1.write('geometric mean : \n')
        f1.write(str(parameter_set_geometric_mean_ratio) + " \n")

        f1.write('best fitness ' + "\n")
        f1.write(str(best_fitness) + '\n')

        best_fitness, Max_energy_fitness_contribution, Localization_duration_ratio_contribution, First_peak_duration_contribution = fit_func1(
            First_peak_Time_duration_best_param,
            max_energy_change_best_param,
            Localization_duration_ratio_best_param,
            parameter_set_geometric_mean_ratio)
        f1.write(
            "contribution from  1. Max energy  2. localization duration ratio  3. first peak duartion:   " + "\n")
        f1.write(str(Max_energy_fitness_contribution) + "  ,  " + str(
            Localization_duration_ratio_contribution) + "  , " + str(First_peak_duration_contribution) + "\n")

        f1.write('First_peak_Time_duration, max_energy_change, Localization_duration ' + "\n")

        f1.write(str(First_peak_Time_duration_best_param) + "  " + str(max_energy_change_best_param) + "  " + str(
            Localization_duration_ratio_best_param) + "\n")

        f1.write('Time: ' + '\n')

        for t in Time_list:
            f1.write(str(t) + " ")
        f1.write('\n')

        f1.write('el:  ' + '\n')
        for energy in d1_energy_list_change:
            f1.write(str(round(energy, 4)) + " ")
        f1.write('\n')

        f1.write('er:  ' + '\n')
        for energy in d2_energy_list_change:
            f1.write(str(round(energy, 4)) + "  ")
        f1.write('\n')

        f1.write('e_photon' + "\n")
        for energy in photon_energy_list:
            f1.write(str(round(energy, 4)) + "  ")
        f1.write('\n')

def Evaluate_simulation_result( *args ):
    ga, full_system_instance , file_path , population_size, coupling_strength , parameter_number  = args

    # ------------------  Evaluate result of Genetic algorithm ----------------------

    # output last generations and their fitfunction.
    Last_generation = ga.last_generation()
    # (member.fitness, member.genes)
    Last_generation_fitness_func = [member[0] for member in Last_generation]
    Last_generation = ga.last_generation()
    Last_generation_param = [member[1] for member in Last_generation]

    # Broadcast fitness function and paramter to all process
    parameter_for_all  = Broadcast_data(Last_generation_param , num_proc )
    fitness_for_all = Broadcast_data( np.real(Last_generation_fitness_func) , num_proc )


    if rank == 0:

        print('num of proc: ' + str(num_proc))
        print('population size: ' + str(population_size))

        best_index = np.argmax(fitness_for_all)
        best_fitness = fitness_for_all[best_index]

        best_param = parameter_for_all[best_index]

        # Sort parameter set according to their fitness function.
        Sort_index = np.argsort(-fitness_for_all)
        parameter_for_all = parameter_for_all[Sort_index]
        fitness_for_all = fitness_for_all[Sort_index]

        # plot best parameter simulation result
        photon_energy_list, d1_energy_list_change, d2_energy_list_change, Time_list = Evolve_full_system_and_return_energy_change(
            full_system_instance, best_param)

        First_peak_Time_duration_best_param, max_energy_change_best_param, Localization_duration_ratio_best_param, localization_bool = Analyze_peak_and_peak_duration(
            d1_energy_list_change, d2_energy_list_change, Time_list)

        # plot simulation result
        plot_simulation_result(Time_list , photon_energy_list, d1_energy_list_change, d2_energy_list_change , file_path )

        # Now you have to save result to files:
        save_simulation_result(    best_param , best_fitness, parameter_number, coupling_strength, Time_list,
            First_peak_Time_duration_best_param , max_energy_change_best_param , Localization_duration_ratio_best_param ,
            d1_energy_list_change , d2_energy_list_change , photon_energy_list , file_path)

def Implement_genetic_algorithm(file_path):
    # - -------  specify input paramter for detector and full system  ----------------

    # --------------- parameter for photon --------------------
    coupling_strength = 0.01
    # fixme : What is photonic energy scale?
    photon_energy_scale = 1

    photon_energy = 1 * photon_energy_scale
    initial_photon_wavefunction = [1/np.sqrt(2) , 1/np.sqrt(2)]

    # ----------- parameter for detector ---------------------------
    dof = 3
    frequency1 = [1, 0.5 , 0.25 ]
    frequency2 = [1, 0.5 , 0.25 ]
    frequency1 = np.array(frequency1) * photon_energy_scale
    frequency2 = np.array(frequency2) * photon_energy_scale
    # fixme : anyway to simplify parameter displamer? Why namx1, nmax2?
    nmax1 = [1, 2 , 4]
    nmax2 = [1, 2, 4 ]

    # fixme : better parameter name ?
    initial_state1 = [0,0,0]
    initial_state2 = [0,0,0]

    # energy window for detector 1 and detector 2 to include state
    energy_window1 = 1 * photon_energy_scale
    energy_window2 = 1 * photon_energy_scale

    # fixme : What is energy window for coupling ?
    # energy window for coupling of states in detector 1 and detector 2.
    energy_window_for_coupling_for_detector1 = 0
    energy_window_for_coupling_for_detector2 = 0

    full_system_energy_window = 0

    Detector_1_parameter = dof, frequency1, nmax1, initial_state1, energy_window1 , energy_window_for_coupling_for_detector1
    Detector_2_parameter = dof, frequency2, nmax2, initial_state2, energy_window2 , energy_window_for_coupling_for_detector2
    # -------------------- end setting up parameter for detector

    # ------------------------------------- End specify parameter for full system and photon ----------

    # ------- declare variable for Genetic algorithm simulation ---------------------
    population_size_over_all_process = 100

    # divide population among differnt process and solve-problem by multiple process.
    population_size = int(population_size_over_all_process / num_proc)

    # parameter for genetic algorithm. See  https://www.wikiwand.com/en/Genetic_algorithm

    generations = 20
    crossover_prob = 0.7
    mutation_prob = 0.01
    # ratio of population to be immigrated between population in different process
    Immigration_ratio = 0.2
    # rate of immigration in genetic algorithm.
    Immigration_rate = 0.1

    #  Initialize full system and construct first part of Hamiltonian.
    full_system_instance = full_system(Detector_1_parameter, Detector_2_parameter, full_system_energy_window, photon_energy, initial_photon_wavefunction )
    full_system_instance.construct_full_system_Hamiltonian_part1()

    if(rank == 0):
    # print information about structure of system
        full_system_instance.print_state_mode()
        full_system_instance.detector1.output_detector_state_coupling()
        full_system_instance.output_off_diagonal_coupling_mode_info()
        print( "parameter number for detector1: "  + str(full_system_instance.detector1.offdiag_coupling_num) )
        print( "parameter number for detector2: " + str(full_system_instance.detector2.offdiag_coupling_num) )
        print( "paramter number for coupling betweeen detector and system:  " + str(full_system_instance.offdiag_param_num -
                                                                                    full_system_instance.detector1.offdiag_coupling_num -
                                                                                    full_system_instance.detector2.offdiag_coupling_num))


    parameter_number = full_system_instance.output_offdiagonal_parameter_number()

    # Prepare seed_data for genetic algorithm
    data = [coupling_strength , full_system_instance, parameter_number]

    # --------------------------------- End  prepare variable for Genetic algorithm simulation -------------------

    # file
    f2 = 0

    if(rank == 0):
        filename = 'info.txt'
        filename = os.path.join(file_path,filename)
        f2 = open(filename, 'w')
        f2.write('population_size:  ' + str(population_size) + '\n')
        f2.write('population_size_in_all_process:  ' + str(population_size_over_all_process) + '\n')
        f2.write('number of process:  ' + str(num_proc) + '\n')

    # ---------------------- initialize Genetic algorithm and do Genetic algorithm ---------------------------------------

    ga  = Extend_Genetic_algorithm(data, population_size = population_size, generations = generations, crossover_probability = crossover_prob, mutation_probability = mutation_prob,
                                    elitism = True, maximise_fitness = True, immigration_ratio= Immigration_ratio , immigrantion_rate= Immigration_rate , info_file = f2)

    ga.fitness_function = fitness_function

    ga.run()

    # ----------------- End of Genetic algorithm ----------------------------------------

    if(rank == 0):
        f2.close()

    # Evaluate simulation result.
    Evaluate_simulation_result(ga, full_system_instance , file_path , population_size, coupling_strength , parameter_number)





