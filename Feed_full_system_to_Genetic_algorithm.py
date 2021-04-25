import numpy as np
import config
from Full_system_class import full_system

from Genetic_algorithm_class import Extend_Genetic_algorithm
from Fitness_function import fitness_function

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()
import os

Time_duration = 5000
output_time_step = 10

config.Time_duration = Time_duration
config.output_time_step = output_time_step

def Implement_genetic_algorithm(file_path):
    # specify input paramter
    coupling_strength = 0.01

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

    Detector_1_parameter = dof, frequency1, nmax1, initial_state1, energy_window1
    Detector_2_parameter = dof, frequency2, nmax2, initial_state2, energy_window2

    Initial_Wavefunction = [1/np.sqrt(2) , 1/np.sqrt(2)]

    population_size_over_all_process = 100
    # Each process only do their part of work. Thus their population is population_size / num_proc
    population_size = int(population_size_over_all_process / num_proc)

    generations = 100
    crossover_prob = 0.7
    mutation_prob = 0.01
    Immigration_ratio = 0.2
    Immigration_frequency = 0.1

    # Other part of code for full system is called within fitness function in each cycle of genetic algorithm.
    full_system_instance = full_system(Detector_1_parameter, Detector_2_parameter, photon_energy, Initial_Wavefunction)
    full_system_instance.construct_full_system_Hamiltonian_part1()

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

