import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()

from Full_system_class import full_system

def Convert_bit_to_parameter( bit_array , delta, parameter_number, bit_per_parameter ):
    '''
    :param: bit_array:  array of bit we have to convert to parameter set
    :param: delta: maximum range of parameter
    :param parameter_number: number of parameter we have to convert
    :param bit_per_parameter: each paramter, we have several bits
        Bit form is the one we used for Genetic algorithm.
    When do simulation, we convert bit to parameters
    :return: parameter set
    '''

    if(len(bit_array) != parameter_number * bit_per_parameter):
        raise NameError(' length of bit_array does not equal to parameter_number * bit_per_parameter ')

    parameter_set = []
    for n in range(parameter_number):
        Bit = bit_array[ n*bit_per_parameter : (n+1) * bit_per_parameter ]
        parameter = 0
        for j in Bit :
            parameter = parameter * 2 + j

        parameter = parameter * delta

        parameter = parameter / pow(2,bit_per_parameter)

        parameter_set.append(parameter)

    return parameter_set

def Convert_parameter_to_bit (parameter_set, delta, parameter_number, bit_per_parameter):
    if(len(parameter_set) != parameter_number):
        raise NameError('length of parameter set is not equal  to parameter number')

    Bit_array = []

    for i in range(parameter_number):
        parameter = parameter_set[i]
        Number = int(np.floor(parameter/ delta * pow(2,bit_per_parameter) ))

        # convert number to bit
        Bit = []
        for j in range(bit_per_parameter):
            Bit.append(  int(Number / pow(2,bit_per_parameter - j - 1)) )
            Number = Number % pow(2,bit_per_parameter - j - 1 )

        Bit_array = Bit_array + Bit

    return Bit_array


def fitness_function(bit_array , data ):
     # data is maximum range of coupling strength
     coupling_strength , full_system_instance, bit_per_parameter, parameter_number = data

     # This code is just used to fake we already implement instance
     full_system_instance = full_system(0,0,1)

     off_diagonal_coupling = Convert_bit_to_parameter(bit_array, coupling_strength, parameter_number, bit_per_parameter)

     # construct off diagonal part of Hamiltonian. and initialize wave function
     full_system_instance.construct_full_system_Hamiltonian_part2(off_diagonal_coupling)


    # Evolve dynamics of full system
     photon_energy_list, d1_energy_list, d2_energy_list, Time_list = full_system_instance.Evolve_dynamics()

    # Analyze_simulation result of full_system. (wave function)


    # return fitness_function