import numpy as np
import matplotlib.pyplot as plt

from include.full_system_class.Full_system_class import full_system

from include.genetic_algorithm_class.Genetic_algorithm_class import Extend_Genetic_algorithm
from Fitness_function import fitness_function , simulate_full_system_energy_flow, Analyze_peak_and_peak_duration , fit_func1 , compute_coupling_geometric_mean

from include.util import Broadcast_data
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()
import os


def Implement_genetic_algorithm(file_path):
    coupling_parameter_range = 0.1
    highest_peak_bool = True

    # --------------- parameter for photon ------------
    photon_energy = 1
    initial_photon_wavefunction = [1/np.sqrt(2) , 1/np.sqrt(2)]

    # ---- parameter for detector ---------
    Detector_1_parameter, Detector_2_parameter = set_detector_param()

    # - -------  specify input paramter for full system  (photon + detector)  ----------------
    # include state in full system (photon + detector) which satisfy energy condition  E - E_init <= energy window
    full_system_energy_window = 0

    # ------------------------------------- End specify parameter for full system and photon & detector----------

    # ------- declare variable for Genetic algorithm simulation ---------------------
    population_size_over_all_process = 100

    # divide population among differnt process and use parallel computing to speed up computation
    population_size = int(population_size_over_all_process / num_proc)

    # parameter for genetic algorithm. See  https://www.wikiwand.com/en/Genetic_algorithm
    generations = 20
    crossover_prob = 0.7
    mutation_prob = 0.01
    # ratio of population to be immigrated between population in different process
    immigrate_population_ratio = 0.2
    # rate of immigration in genetic algorithm.
    immigration_rate = 0.1

    #  Initialize full system and construct first part of Hamiltonian.
    full_system_instance = full_system(Detector_1_parameter, Detector_2_parameter, full_system_energy_window, photon_energy, initial_photon_wavefunction )
    full_system_instance.construct_full_system_Hamiltonian_part1()

    # print information about structure of system
    if(rank == 0):
        output_full_system_state_and_coupling_info(full_system_instance)

    # parameter_number is number of parameter to be optimized in Genetic algorithm.
    parameter_number = full_system_instance.output_offdiagonal_parameter_number()

    # Prepare parameter for genetic algorithm
    param = [coupling_parameter_range , full_system_instance, parameter_number , highest_peak_bool]

    # --------------------------------- End  prepare variable for Genetic algorithm simulation -------------------

    # info file for Genetic algorithm.
    f2 = genetic_algorithm_info(file_path, population_size, population_size_over_all_process)

    # ---------------------- initialize Genetic algorithm and do Genetic algorithm ---------------------------------------
    ga  = Extend_Genetic_algorithm(param, population_size = population_size, generations = generations, crossover_probability = crossover_prob, mutation_probability = mutation_prob,
                                    elitism = True, maximise_fitness = True, immigration_ratio= immigrate_population_ratio , immigrantion_rate= immigration_rate , info_file = f2)

    # customized fitness function for genetic algorithm.
    ga.fitness_function = fitness_function

    ga.run()

    # ----------------- End of Genetic algorithm ----------------------------------------
    if(rank == 0):
        f2.close()

    # Evaluate simulation result.
    Evaluate_simulation_result(ga, full_system_instance , file_path , population_size, coupling_parameter_range , parameter_number , highest_peak_bool)

def set_detector_param():
    # ----------- parameter for detector ---------------------------
    dof = 3
    # -------------- parameter for d1 -------------------------
    # frequency of detector's vib mode
    frequency1 = np.array([1, 0.5, 0.25])
    nmax1 = [1, 2, 4]
    initial_d_state1 = [0, 0, 0]
    # energy window for detector to include state around init state.
    d1_energy_window = 1
    # state couple to each other in detector should satisfy \Delta E <= state_coupling_energy_window.
    # set it == 0 means only resonant state can couple to each other.
    d1_state_coupling_energy_window = 0

    # -------------- parameter for d2 ----------------
    frequency2 = np.array([1, 0.5, 0.25])
    nmax2 = [1, 2, 4]
    initial_d_state2 = [0, 0, 0]
    d2_energy_window = 1
    d2_state_coupling_energy_window = 0

    Detector_1_parameter = dof, frequency1, nmax1, initial_d_state1, d1_energy_window, d1_state_coupling_energy_window
    Detector_2_parameter = dof, frequency2, nmax2, initial_d_state2, d2_energy_window, d2_state_coupling_energy_window
    # -------------------- end setting up parameter for detector ----------------------------
    return Detector_1_parameter, Detector_2_parameter

def output_full_system_state_and_coupling_info(full_system_instance):
    if(rank == 0):
        # print information about structure of system
        full_system_instance.output_state_mode()
        full_system_instance.detector1.output_detector_state_coupling()
        full_system_instance.output_off_diagonal_coupling_mode_info()
        print("parameter number for detector1: " + str(full_system_instance.detector1.offdiag_coupling_num))
        print("parameter number for detector2: " + str(full_system_instance.detector2.offdiag_coupling_num))
        print("paramter number for coupling betweeen detector and system:  " + str(full_system_instance.offdiag_param_num -
                                                                                   full_system_instance.detector1.offdiag_coupling_num -
                                                                                   full_system_instance.detector2.offdiag_coupling_num))

def genetic_algorithm_info(file_path, population_size, population_size_over_all_process):
    # info file for Genetic algorithm.
    f2 = 0

    if (rank == 0):
        filename = 'info.txt'
        filename = os.path.join(file_path, filename)
        f2 = open(filename, 'w')
        f2.write('population_size:  ' + str(population_size) + '\n')
        f2.write('population_size_in_all_process:  ' + str(population_size_over_all_process) + '\n')
        f2.write('number of process:  ' + str(num_proc) + '\n')

    return f2

def Evaluate_simulation_result( *args ):
    ga, full_system_instance , file_path , population_size, coupling_strength , parameter_number, highest_peak_bool = args

    # ------------------  Evaluate result of Genetic algorithm ----------------------

    # output last generations and their fitfunction.
    last_generation = ga.last_generation()
    # (member.fitness, member.genes)
    last_generation_fitness_func = [member[0] for member in last_generation]
    last_generation_param = [member[1] for member in last_generation]

    # Broadcast fitness function and paramter to all process
    parameter_for_all = Broadcast_data(last_generation_param , num_proc )
    fitness_for_all = Broadcast_data( np.real(last_generation_fitness_func) , num_proc )


    if rank == 0:

        print('num of proc: ' + str(num_proc))
        print('population size: ' + str(population_size))

        best_gene_index = np.argmax(fitness_for_all)
        best_fitness = fitness_for_all[best_gene_index]
        best_param = parameter_for_all[best_gene_index]

        # Sort parameter set according to their fitness function.
        sort_index = np.argsort(-fitness_for_all)
        parameter_for_all = parameter_for_all[sort_index]
        fitness_for_all = fitness_for_all[sort_index]

        # plot best parameter simulation result
        photon_energy_list, d1_energy_list_change, d2_energy_list_change, time_list = simulate_full_system_energy_flow(
            full_system_instance, best_param)

        # plot simulation result
        plot_simulation_result( photon_energy_list, d1_energy_list_change, d2_energy_list_change , time_list , file_path )

        # analyze result
        analyze_result = \
            Analyze_peak_and_peak_duration( d1_energy_list_change, d2_energy_list_change, time_list , highest_peak_bool= highest_peak_bool)

        # Now you have to save result to files:
        save_simulation_result(  best_param , best_fitness, parameter_number, coupling_strength, time_list,
            analyze_result,
            d1_energy_list_change , d2_energy_list_change , photon_energy_list , file_path)

def plot_simulation_result(*args):
    photon_energy_list, d1_energy_list_change, d2_energy_list_change, Time_list, file_path = args
    # configure fig, ax
    fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.plot(Time_list, d1_energy_list_change, label='left photon localization')
    ax1.plot(Time_list, d2_energy_list_change, label='right photon localization')
    ax1.plot(Time_list, photon_energy_list, label='photon energy')

    ax1.set_xlabel('time')
    ax1.set_ylabel('E')

    ax1.legend(loc='best')

    # save figure.
    fig_name = "best_simulation_result.png"
    fig_name = os.path.join(file_path, fig_name)

    fig1.savefig(fig_name)

def save_simulation_result(*args):
    best_param, best_fitness, parameter_number, parameter_range, Time_list, \
    analyze_result, \
    d1_energy_list_change, d2_energy_list_change, photon_energy_list, file_path = args

    first_peak_time_duration_ratio_best_param, max_energy_change_best_param, localization_duration_ratio_best_param, _ = analyze_result

    # save best simulation result found
    filename1 = 'best_parameter_and_fitness_function.txt'
    filename1 = os.path.join(file_path, filename1)
    with open(filename1, 'w') as f1:
        # record parameter we fit.
        for num in best_param:
            f1.write(str(num) + " , ")
        f1.write('\n')

        # compute geometric mean and write to output file.
        normalized_geometric_mean = compute_coupling_geometric_mean(best_param, parameter_number, parameter_range)

        f1.write('geometric mean : \n')
        f1.write(str(normalized_geometric_mean) + " \n")

        f1.write('best fitness ' + "\n")
        f1.write(str(best_fitness) + '\n')

        # fit function
        best_fitness_by_fitting, max_energy_fitness_contribution, localization_duration_ratio_contribution, first_peak_duration_contribution = fit_func1(
            first_peak_time_duration_ratio_best_param,
            max_energy_change_best_param,
            localization_duration_ratio_best_param,
            normalized_geometric_mean)

        # assert the fitness function we get is correct.
        fitness_func_diff = np.abs(best_fitness - best_fitness_by_fitting) / np.abs(best_fitness)
        assert (fitness_func_diff < 0.05)

        f1.write(
            "contribution from  1. Max energy  2. localization duration ratio  3. first peak duartion_ratio:   " + "\n")
        f1.write(str(max_energy_fitness_contribution) + "  ,  " + str(
            localization_duration_ratio_contribution) + "  , " + str(first_peak_duration_contribution) + "\n")

        f1.write('first_peak_Time_duration_ratio, max_energy_change, localization_duration ' + "\n")

        f1.write(str(first_peak_time_duration_ratio_best_param) + "  " + str(max_energy_change_best_param) + "  " + str(
            localization_duration_ratio_best_param) + "\n")

        # record : time, el, er, energy.
        write_data(f1, Time_list, 'Time: ')
        write_data(f1, d1_energy_list_change, "el:  ")
        write_data(f1, d2_energy_list_change, "er:  ")
        write_data(f1, photon_energy_list, "e_photon:  ")

def write_data(f, data_list, symbol):
    f.write(symbol + "\n")
    for data in data_list:
        f.write(str(round(data, 4)) + " ")
    f.write("\n")



