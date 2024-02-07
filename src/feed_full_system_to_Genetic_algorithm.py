import numpy as np

import include.fullsystem.__init__

from include.ExtendedGeneticAlgorithm import ExtendedGeneticAlgorithm
import simulate_energy_flow_between_photon_and_detector
import include.util
from mpi4py import MPI

import output_simulation_result
import set_simulation_input_param

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()


def implement_genetic_algorithm(file_path):
    coupling_parameter_range = 0.01  # the coupling range decides the timescale for the localization

    highest_peak_bool = True # fixme: need explanation

    # --------------- parameter for photon ------------
    photon_energy = 1
    initial_photon_wavefunction = [1/np.sqrt(2) , 1/np.sqrt(2)]

    # ---- parameter for detector ---------
    detector_1_parameter, detector_2_parameter = set_simulation_input_param.set_detector_param()

    # - -------  specify input paramter for full system  (photon + detector)  ----------------
    # include states in full system (photon + detector) which satisfy energy condition  E - E_init <= energy window
    full_system_basis_set_energy_window = 0

    # ------------------------------------- End specify parameter for full system and photon & detector----------

    # ------- declare variable for Genetic algorithm simulation ---------------------
    population_size_over_all_process = 1000

    # divide population among different processes and use parallel computing to speed up computation
    population_size_each_process = int(population_size_over_all_process / num_proc)

    # parameter for genetic algorithm. See  https://www.wikiwand.com/en/Genetic_algorithm
    generations = 20
    crossover_prob = 0.7
    mutation_prob = 0.01
    # ratio of population to be immigrated between population in different process
    immigrate_population_ratio = 0.2
    # rate of immigration in genetic algorithm.
    immigration_rate = 0.1

    # Time for evolving wave function to calculate the energy transfer between photon and detectors.
    time_duration = 5000  # duration for evolution of the wave function
    output_time_step = 10 # time step to compute photon energy and detectors' energy.

    #  Initialize full system and construct first part of Hamiltonian.
    full_system_instance = include.fullsystem.__init__.FullSystem(detector_1_parameter, detector_2_parameter, full_system_basis_set_energy_window,
                                      photon_energy, initial_photon_wavefunction,
                                      time_duration = time_duration, output_time_step = output_time_step)

    full_system_instance.construct_full_system_hamiltonian_structure()

    # print information about structure of system
    if rank == 0:
        output_simulation_result.output_full_system_state_and_coupling_info(full_system_instance)

    # parameter_number is number of off-diagonal coupling parameter numbers to be optimized in Genetic algorithm.
    parameter_number  =  full_system_instance.show_offdiag_matrix_num()

    # Prepare parameter for genetic algorithm
    parameter_list = [coupling_parameter_range , full_system_instance, parameter_number , highest_peak_bool]
    # --------------------------------- End  prepare variable for Genetic algorithm simulation -------------------

    # prepare info file for Genetic algorithm.
    f2 = set_simulation_input_param.prepare_genetic_algorithm_info_file(file_path, population_size_each_process, population_size_over_all_process)

    # ---------------------- initialize Genetic algorithm and drun Genetic algorithm. ---------------------------------------
    ga  = ExtendedGeneticAlgorithm.ExtendGeneticAlgorithm(parameter_list, population_size = population_size_each_process, generations = generations, crossover_probability = crossover_prob, mutation_probability = mutation_prob,
                                 elitism = True, maximise_fitness = True, immigration_population_ratio= immigrate_population_ratio, immigrantion_rate= immigration_rate, info_file = f2)

    ga.run()

    # ----------------- End of Genetic algorithm ----------------------------------------
    if rank == 0:
        f2.close() # close information file.

    # Evaluate simulation result.
    evaluate_simulation_result(ga, full_system_instance, file_path, population_size_each_process, coupling_parameter_range, parameter_number, highest_peak_bool)


def evaluate_simulation_result(*args):
    '''
    Evaluate the result of the genetic algorithm.
    :param args:
    :return:
    '''
    ga, full_system_instance , file_path , population_size, coupling_strength , parameter_number, highest_peak_bool = args

    # output last generations and their fitfunction.  (chromosome type)
    last_generation = ga.show_current_generation()

    # (member.fitness, member.genes). chromosome variable in last_generation.
    last_generation_fitness_func = [member.fitness for member in last_generation]
    last_generation_gene = [member.genes for member in last_generation]

    # Broadcast fitness function and paramter to all process
    genes_for_all = include.util.gather_and_broadcast_data(last_generation_gene, num_proc)
    fitness_for_all = include.util.gather_and_broadcast_data(np.real(last_generation_fitness_func), num_proc)

    if rank == 0:
        print('num of proc: ' + str(num_proc))
        print('population size: ' + str(population_size))

        best_gene_index = np.argmax(fitness_for_all)
        best_fitness = fitness_for_all[best_gene_index]
        best_genes = genes_for_all[best_gene_index]

        # Sort parameter set in descending order according to their fitness function.
        sort_index = np.argsort(-fitness_for_all)
        genes_for_all = genes_for_all[sort_index]
        fitness_for_all = fitness_for_all[sort_index]

        # plot best parameter simulation result
        photon_energy_list, detector1_energy_list_change, detector2_energy_list_change, time_list = simulate_energy_flow_between_photon_and_detector.simulate_full_system_quantum_dynamics(
            full_system_instance, best_genes)

        # plot simulation result
        output_simulation_result.plot_simulation_result(photon_energy_list, detector1_energy_list_change,
                               detector2_energy_list_change, time_list, file_path)

        # analyze result
        analyze_result = \
            simulate_energy_flow_between_photon_and_detector.analyze_peak_and_peak_duration(detector1_energy_list_change, detector2_energy_list_change,
                                           time_list, highest_peak_bool= highest_peak_bool)

        # Now you have to save result to files:
        output_simulation_result.save_simulation_result(best_genes, best_fitness, parameter_number, coupling_strength, time_list,
                               analyze_result,
                               detector1_energy_list_change, detector2_energy_list_change, photon_energy_list, file_path)



