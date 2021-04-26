import numpy as np
import matplotlib.pyplot as plt
from Full_system_class import full_system

from Genetic_algorithm_class import Extend_Genetic_algorithm
from Fitness_function import fitness_function , Evolve_full_system_and_return_energy_change, Analyze_peak_and_peak_duration
from Fitness_function import Convert_bit_to_parameter, fit_func1

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()
import os


def Implement_genetic_algorithm(file_path):
    # specify input paramter
    coupling_strength = 0.1

    photon_energy = 1

    dof = 3
    frequency1 = [1, 0.5 , 0.25 ]
    frequency2 = [1, 0.5 , 0.25 ]

    nmax1 = [1, 2, 4]
    nmax2 = [1, 2 , 4]

    initial_state1 = [0,0,0]
    initial_state2 = [0,0,0]

    energy_window1 = 1
    energy_window2 = 1

    full_system_energy_window = 0

    Detector_1_parameter = dof, frequency1, nmax1, initial_state1, energy_window1
    Detector_2_parameter = dof, frequency2, nmax2, initial_state2, energy_window2

    Initial_Wavefunction = [1/np.sqrt(2) , 1/np.sqrt(2)]

    population_size_over_all_process = 1000
    # Each process only do their part of work. Thus their population is population_size / num_proc
    population_size = int(population_size_over_all_process / num_proc)

    generations = 20
    crossover_prob = 0.7
    mutation_prob = 0.01
    Immigration_ratio = 0.2
    Immigration_frequency = 0.1

    # Other part of code for full system is called within fitness function in each cycle of genetic algorithm.
    full_system_instance = full_system(Detector_1_parameter, Detector_2_parameter, full_system_energy_window, photon_energy, Initial_Wavefunction)
    full_system_instance.construct_full_system_Hamiltonian_part1()

    # print information about structure of system
    full_system_instance.print_state_mode()
    full_system_instance.detector1.output_detector_state_coupling()

    parameter_number = full_system_instance.output_offdiagonal_parameter_number()
    bit_per_parameter = 7

    # Prepare seed_data for genetic algorithm
    data = [coupling_strength , full_system_instance, bit_per_parameter, parameter_number]

    # file
    f2 = 0

    if(rank == 0):
        filename = 'info.txt'
        filename = os.path.join(file_path,filename)
        f2 = open(filename, 'w')
        f2.write('population_size:  ' + str(population_size) + '\n')
        f2.write('population_size_in_all_process  ' + str(population_size_over_all_process) + '\n')
        f2.write('number of process  ' + str(num_proc) + '\n')

    ga  = Extend_Genetic_algorithm(data, population_size = population_size, generations = generations, crossover_probability = crossover_prob, mutation_probability = mutation_prob,
                                    elitism = True, maximise_fitness = True, immigration_ratio= Immigration_ratio , immigrantion_frequency= Immigration_frequency , info_file = f2)

    ga.fitness_function = fitness_function

    ga.run()

    if(rank == 0):
        f2.close()


    # Evaluate result:

    # output last generations and their fitfunction.
    Last_generation = ga.last_generation()
    # (member.fitness, member.genes)
    Last_generation_fitness_func = [member[0] for member in Last_generation]
    Last_generation = ga.last_generation()
    Last_generation_bit_array = [member[1] for member in Last_generation]


    Parameter_set = []
    for i in range(population_size):
        bit = Last_generation_bit_array[i]
        parameter = Convert_bit_to_parameter(bit, coupling_strength, parameter_number,bit_per_parameter )
        Parameter_set.append(parameter)
    # use MPI to send Last_generation_fitness_func and Last_generation_bit_array

    # prepare sending data
    send_param = np.array(Parameter_set)
    send_fitness = np.array(np.real(Last_generation_fitness_func))

    # paraent process prepare space to receive data

    # for children process, we also have to specify recv_bit and recv_fitness
    recv_param = np.zeros([0])
    recv_fitness = np.zeros([0])

    if rank == 0:
        # parent process
        print('num of proc: ' + str(num_proc))
        print('population size: ' + str(population_size))

        recv_param = np.empty([num_proc, population_size, parameter_number], dtype=np.float64)
        recv_fitness = np.empty([num_proc, population_size], dtype=np.float64)

    comm.Gather(send_param, recv_param, 0)
    comm.Gather(send_fitness, recv_fitness, 0)

    if rank == 0:
        # Now we have all final genetic algorithm simulation result.
        # recv_bit : [ num_proc, maximum_population_in_each_process, Bit_number ] contain information about parameter
        # recv_fitness: [num_proc, maximum_population_in_each_process] contain information about fitness function

        fitness_for_all = np.reshape(recv_fitness, (recv_fitness.shape[0] * recv_fitness.shape[1]) )
        parameter_for_all = np.reshape(recv_param , (recv_param.shape[0] * recv_param.shape[1] , recv_param.shape[2]) )

        best_index = np.argmax(fitness_for_all)
        best_fitness = fitness_for_all[best_index]

        best_param = parameter_for_all[best_index]

        # Sort parameter set according to their fitness function.
        Sort_index = np.argsort( -fitness_for_all)
        parameter_for_all = parameter_for_all[Sort_index]
        fitness_for_all = fitness_for_all[Sort_index]

        # plot best parameter simulation result
        full_system_instance.construct_full_system_Hamiltonian_part2(best_param)
        photon_energy_list, d1_energy_list_change, d2_energy_list_change, Time_list = Evolve_full_system_and_return_energy_change(full_system_instance)

        First_peak_Time_duration, max_energy_change, Localization_duration_ratio = Analyze_peak_and_peak_duration(
            d1_energy_list_change, d2_energy_list_change, Time_list)

        # plot simulation result
        fig1, ax1 = plt.subplots(nrows=1, ncols=1)
        ax1.plot(Time_list, d1_energy_list_change , label='left photon localization')
        ax1.plot(Time_list , d2_energy_list_change , label='right photon localization')
        ax1.plot(Time_list , photon_energy_list , label='photon energy')

        ax1.legend(loc = 'best')

        fig_name = "best_simulation_result.png"
        fig_name = os.path.join(file_path, fig_name)

        fig1.savefig(fig_name)

        # Now you have to save result to files:
        filename = 'parameter_set_last_generation.txt'
        filename = os.path.join(file_path, filename)
        with open(filename, 'w') as f:
            parameter_list_len = len(parameter_for_all)
            for i in range(parameter_list_len):
                parameter = parameter_for_all[i]

                # write parameter and their fitness function
                for number in parameter:
                    f.write(str(number) + " ")
                f.write('\n')
                f.write(str(fitness_for_all[i]))
                f.write('\n')

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

            best_fitness , Max_energy_fitness_contribution, Localization_duration_ratio_contribution , First_peak_duration_contribution = fit_func1(First_peak_Time_duration,max_energy_change,Localization_duration_ratio,parameter_set_geometric_mean_ratio)
            f1.write("contribution from  1. Max energy  2. localization duration ratio  3. first peak duartion:   " + "\n")
            f1.write(str(Max_energy_fitness_contribution) + "  ,  " + str(Localization_duration_ratio_contribution) + "  , " + str(First_peak_duration_contribution) + "\n"  )

            f1.write('First_peak_Time_duration, max_energy_change, Localization_duration ' + "\n")

            f1.write(str(First_peak_Time_duration) + "  " + str(max_energy_change) + "  " + str(
                Localization_duration_ratio) + "\n")

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





