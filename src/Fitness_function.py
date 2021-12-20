import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()
import include.Shared_data
from include.Full_system_class import full_system

def Analyze_peak_and_peak_duration(e2l_change, e2r_change , Time ) :

    max_e2l_change = max(e2l_change)
    max_e2r_change = max(e2r_change)

    # we find first peak instead of highest peak. We first find highest peak.
    # Then set ratio * highest peak as threashould to be treated as a peak. Then we find earliest peak in simulation.
    max_energy_change = max(max_e2l_change, max_e2r_change)
    ratio = 0.8
    criteria_for_peak = ratio * max_energy_change

    e2l_change_peak_index = [i for i in range(len(e2l_change)) if e2l_change[i] > criteria_for_peak]
    e2r_change_peak_index = [i for i in range(len(e2r_change)) if e2r_change[i] > criteria_for_peak]

    Localization_bool = 0
    if (len(e2l_change_peak_index) == 0):
        Localization_bool = 2
    elif (len(e2r_change_peak_index) == 0):
        Localization_bool = 1
    else:
        if (e2l_change_peak_index[0] < e2r_change_peak_index[0]):
            # first peak appear at left
            Localization_bool = 1

        elif (e2l_change_peak_index[0] > e2r_change_peak_index[0]):
            Localization_bool = 2

    # ------------------  Another criteria -----------------------------
    if(max_energy_change == max_e2l_change ):
        Localization_bool = 1
    else:
        Localization_bool = 2
    # -------------------------------------------------

    time_step = Time[1] - Time[0]
    Localization_duration_left = len(e2l_change_peak_index) * time_step
    Localization_duration_right = len(e2r_change_peak_index) * time_step
    Localization_duration_ratio = 0
    if Localization_bool == 1:
        Localization_duration_ratio = Localization_duration_left / Time[-1]
    else:
        Localization_duration_ratio = Localization_duration_right / Time[-1]

    # first peak list and its index
    e2l_change_first_peak_index = []
    e2r_change_first_peak_index = []

    # value of first peak:
    if len(e2l_change_peak_index) > 1:
        for i in range(1, len(e2l_change_peak_index)):
            e2l_change_first_peak_index.append(e2l_change_peak_index[i - 1])
            if (e2l_change_peak_index[i] - e2l_change_peak_index[i - 1] != 1):
                break
    else:
        if (len(e2l_change_peak_index) == 1):
            e2l_change_first_peak_index.append(e2l_change_peak_index[0])

    if len(e2r_change_peak_index) > 1:
        for i in range(1, len(e2r_change_peak_index)):
            e2r_change_first_peak_index.append(e2r_change_peak_index[i - 1])
            if (e2r_change_peak_index[i] - e2r_change_peak_index[i - 1] != 1):
                break
    else:
        if (len(e2r_change_peak_index) == 1):
            e2r_change_first_peak_index.append(e2r_change_peak_index[0])

    e2l_change_first_peak = e2l_change[e2l_change_first_peak_index]
    e2r_change_first_peak = e2r_change[e2r_change_first_peak_index]

    if (len(e2l_change_first_peak) != 0):
        max_e2l_change = max(e2l_change_first_peak)
    if (len(e2r_change_first_peak) != 0):
        max_e2r_change = max(e2r_change_first_peak)

    First_peak_Time_duration_right = 0
    First_peak_Time_duration_left =  0

    if(len(e2r_change_first_peak_index)  > 0) :
        First_peak_time_right = np.array(Time)[e2r_change_first_peak_index]
        First_peak_list_right = e2r_change[e2r_change_first_peak_index]
        First_peak_Time_duration_right = First_peak_time_right[-1] - First_peak_time_right[0]\

    # now we find peak for left detector
    if( len(e2l_change_first_peak_index) > 0  ):
        First_peak_time_left = np.array(Time)[e2l_change_first_peak_index]
        First_peak_list_left = e2l_change[e2l_change_first_peak_index]
        First_peak_Time_duration_left = First_peak_time_left[-1] - First_peak_time_left[0]

    if Localization_bool == 1 :
        First_peak_Time_duration = First_peak_Time_duration_left
        max_energy_change = max_e2l_change
    else:
        First_peak_Time_duration = First_peak_Time_duration_right
        max_energy_change = max_e2r_change

    return First_peak_Time_duration, max_energy_change,Localization_duration_ratio, Localization_bool

def fit_func1(First_peak_Time_duration, max_energy_change ,  Localization_duration_ratio, parameter_geometric_mean_ratio):
    '''
    see Genetic_algorithm_fitness_func.py : fit_func for requirement of fit_func
    # We need to be very careful when defining fitness function.

    Localization_duration_ratio is in range[0,1], max_energy_change is also in range[0,1]

    parameter_geometric_mean_ratio is geometric mean of parameter / upper limit you set  for parameter.
    This is to make sure we don't sample extremely small parameter to get long localization time.
    :return:

    '''
    # ratio of first peak duration.
    Time_duration = include.Shared_data.Time_duration

    First_peak_Time_duration_ratio = First_peak_Time_duration / Time_duration * parameter_geometric_mean_ratio

    # Fitness_func_value = 0.5 * pow(max_energy_change, 2 ) + 10 * pow(Localization_duration_ratio , 2) + 60 * First_peak_Time_duration_ratio

    Max_energy_fitness_contribution = 5 * pow(max_energy_change, 2 )

    Localization_duration_ratio_contribution = 10 * pow(Localization_duration_ratio , 2)

    First_peak_duration_contribution = First_peak_Time_duration_ratio * 60

    Fitness_func_value = Max_energy_fitness_contribution + Localization_duration_ratio_contribution + First_peak_duration_contribution

    return  Fitness_func_value , Max_energy_fitness_contribution, Localization_duration_ratio_contribution , First_peak_duration_contribution

def Evolve_full_system_and_return_energy_change(full_system_instance , off_diagonal_coupling ):
    # construct off diagonal part for Hamiltonian and re-initialize wave function.
    full_system_instance.construct_full_system_Hamiltonian_part2(off_diagonal_coupling)
    # Evolve dynamics of full system
    photon_energy_list, d1_energy_list, d2_energy_list, Time_list = full_system_instance.Evolve_dynamics()

    # Analyze_simulation result of full_system. (wave function)
    photon_energy = full_system_instance.photon_energy
    photon_energy_list = photon_energy_list / photon_energy
    d1_energy_list_change = d1_energy_list - d1_energy_list[0]
    d2_energy_list_change = d2_energy_list - d2_energy_list[0]

    d1_energy_list_change = d1_energy_list_change / photon_energy
    d2_energy_list_change = d2_energy_list_change / photon_energy

    return photon_energy_list, d1_energy_list_change, d2_energy_list_change, Time_list

def fitness_function(off_diagonal_coupling , data ):
     # data is maximum range of coupling strength
     parameter_range , full_system_instance, parameter_number = data

     # construct off diagonal part of Hamiltonian. and initialize wave function
     photon_energy_list, d1_energy_list_change, d2_energy_list_change, Time_list = Evolve_full_system_and_return_energy_change(full_system_instance , off_diagonal_coupling)

     First_peak_Time_duration, max_energy_change,Localization_duration_ratio, localization_bool = Analyze_peak_and_peak_duration(d1_energy_list_change, d2_energy_list_change, Time_list)

     # compute geometric mean and its ratio to coupling strength.
     geometric_mean = 1
     for i in range(parameter_number):
         geometric_mean = geometric_mean * pow(off_diagonal_coupling[i] , 1/parameter_number)


     geometric_mean_ratio = geometric_mean / parameter_range

     fitness_func_value , Max_energy_fitness_contribution, Localization_duration_ratio_contribution , First_peak_duration_contribution  = fit_func1(First_peak_Time_duration, max_energy_change, Localization_duration_ratio,
                                   geometric_mean_ratio)

     return fitness_func_value