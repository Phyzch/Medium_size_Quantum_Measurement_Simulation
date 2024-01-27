import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()
from include.fullsystem.__init__ import FullSystem

def extract_first_peak(list):
    # extract list of consecutive number from list as set this as first peak.
    list_len = len(list)
    first_peak_list = []
    if list_len > 1:
        for i in range(1, list_len):
            first_peak_list.append(list[i-1])
            if(list[i] - list[i-1] != 1):
                break

    else:
        if(list_len == 1):
            first_peak_list = [ list[0] ]

    return first_peak_list

def analyze_peak_property(time, energy_change, energy_change_peak_index):
    '''
    Three variables are of particular interest:
    1. time duration for first peak
    2. maximum energy change for first peak
    3. time for localization.
    :return : first_peak_time_duration, max_energy_change_first_peak , localization_duration_ratio
    '''
    time_step = time[1] - time[0]
    final_time = time[-1]

    # time to localize
    localization_duration = len(energy_change_peak_index) * time_step
    localization_duration_ratio = localization_duration / final_time

    # first peak's index and peak value:  (here peak is peak for energy change)
    energy_change_first_peak_index = extract_first_peak(energy_change_peak_index)
    first_peak_len = len(energy_change_first_peak_index)

    energy_change_first_peak = energy_change[energy_change_first_peak_index]
    first_peak_time = np.array(time)[energy_change_first_peak_index]
    # time_duration.
    first_peak_time_duration = (first_peak_time[-1] - first_peak_time[0]) if first_peak_len > 0 else 0
    first_peak_time_duration_ratio = first_peak_time_duration / final_time
    # maximum energy change for first peak. We set maximum energy change for first peak and max energy change.
    max_energy_change_first_peak = max(energy_change_first_peak) if first_peak_len !=0 else 0


    return first_peak_time_duration_ratio , max_energy_change_first_peak , localization_duration_ratio

def decide_localization_side(highest_peak_bool, max_energy_change , max_e2l_change , max_e2r_change, e2l_change_peak_index ,  e2r_change_peak_index ):
    if(highest_peak_bool):
        # ------- criteria: use peak's highest energy. ---------------------
        if(max_energy_change == max_e2l_change ):
            localization_bool = 1
        else:
            localization_bool = 2
        # --------------------------------------------------------------
    else:
        # --------- criteria : use earliest peak as criteria ------------
        localization_bool = 0
        if (len(e2l_change_peak_index) == 0):
            localization_bool = 2
        elif (len(e2r_change_peak_index) == 0):
            localization_bool = 1
        else:
            if ( e2l_change_peak_index[0] < e2r_change_peak_index[0]):
                # first peak appear at left
                localization_bool = 1

            elif ( e2l_change_peak_index[0] > e2r_change_peak_index[0]):
                localization_bool = 2

    return localization_bool

def Analyze_peak_and_peak_duration(e2l_change, e2r_change, time , highest_peak_bool ) :
    '''

    :param e2l_change: change of energy for l.h.s.
    :param e2r_change: change of energy for r.h.s.
    :param time:
    :param highest_peak_bool : if bool == True, we choose highest peak as criteria for localization.
                               if bool == False, we choose first peak as criteria for localization.
    :return:
    '''

    max_e2l_change = max(e2l_change)
    max_e2r_change = max(e2r_change)
    max_energy_change = max(max_e2l_change, max_e2r_change)

    # if we use first peak instead of highest peak as criteria. (highest_peak_bool == False)
    # We first find highest peak.
    # Then set ratio * highest peak as threashould to be treated as a peak. Then we find earliest peak in simulation.
    peak_criteria_ratio = 0.7
    criteria_for_peak = peak_criteria_ratio * max_energy_change


    e2l_change_peak_index = [i for i in range(len(e2l_change)) if e2l_change[i] > criteria_for_peak]
    e2r_change_peak_index = [i for i in range(len(e2r_change)) if e2r_change[i] > criteria_for_peak]

    # ----  decide which side this simulation localize ---------
    localization_bool = decide_localization_side(highest_peak_bool, max_energy_change, max_e2l_change, max_e2r_change,
                             e2l_change_peak_index, e2r_change_peak_index)
    # -----------------------------------------

    if localization_bool == 1:
        # localize to left
        first_peak_time_duration_ratio, max_energy_change , localization_duration_ratio = analyze_peak_property(time, e2l_change, e2l_change_peak_index)
    else:
        first_peak_time_duration_ratio, max_energy_change , localization_duration_ratio = analyze_peak_property(time, e2r_change, e2r_change_peak_index)


    return first_peak_time_duration_ratio, max_energy_change,localization_duration_ratio, localization_bool

def simulate_full_system_energy_flow(full_system_instance, off_diagonal_coupling):
    # construct off diagonal part for Hamiltonian
    full_system_instance.construct_full_system_Hamiltonian_part2(off_diagonal_coupling)

    # initialize wave function
    full_system_instance.initialize_wave_function()

    # Evolve dynamics of full system
    photon_energy_list, d1_energy_list, d2_energy_list, time_list = full_system_instance.Evolve_dynamics()

    # Analyze_simulation result of full_system. (wave function)
    init_photon_energy = full_system_instance.init_photon_energy

    d1_energy_list_change = d1_energy_list - d1_energy_list[0]
    d2_energy_list_change = d2_energy_list - d2_energy_list[0]

    photon_energy_list = photon_energy_list / init_photon_energy
    d1_energy_list_change = d1_energy_list_change / init_photon_energy
    d2_energy_list_change = d2_energy_list_change / init_photon_energy

    return photon_energy_list, d1_energy_list_change, d2_energy_list_change, time_list


def fitness_function(off_diagonal_coupling , param  ):
     # data is maximum range of coupling strength
     parameter_range , full_system_instance, parameter_number , highest_peak_bool  = param

     # construct off diagonal part of Hamiltonian. and initialize wave function
     photon_energy_list, d1_energy_list_change, d2_energy_list_change, time_list = simulate_full_system_energy_flow( full_system_instance, off_diagonal_coupling)

     first_peak_time_duration_ratio, max_energy_change, localization_duration_ratio, localization_bool = \
         Analyze_peak_and_peak_duration(d1_energy_list_change, d2_energy_list_change, time_list, highest_peak_bool= highest_peak_bool )

     # compute geometric mean and its ratio to coupling strength.
     normalized_geometric_mean = compute_coupling_geometric_mean(off_diagonal_coupling , parameter_number , parameter_range)

     fitness_func_value , Max_energy_fitness_contribution, Localization_duration_ratio_contribution , First_peak_duration_contribution  = fit_func1(first_peak_time_duration_ratio , max_energy_change, localization_duration_ratio,
                                   normalized_geometric_mean)

     return fitness_func_value

def compute_coupling_geometric_mean(off_diagonal_coupling , parameter_number, parameter_range):
    # pow(  \prod coupling / parameter_range , 1/N)
    geometric_mean = 1
    for i in range(parameter_number):
        geometric_mean = geometric_mean * pow(off_diagonal_coupling[i], 1 / parameter_number)
    normalized_geometric_mean = geometric_mean / parameter_range

    return normalized_geometric_mean

def fit_func1(first_peak_Time_duration_ratio, max_energy_change, Localization_duration_ratio, parameter_geometric_mean_ratio):
    '''
    We need to be very careful when defining fitness function.

    localization_duration_ratio is in range[0,1], max_energy_change is in range[0,1],  first_peak_time_duration_ratio is also in range[0,1]

    parameter_geometric_mean_ratio is geometric mean of parameter / upper limit you set  for parameter.
    This is to make sure we don't sample extremely small parameter to get long localization time.

    This choice of fitness function is tricky and ad-hoc, try different choices to see which works best.
    See p.97 in Practical Genetic algorithms (R.L.Haupt & S.E.Haupt) about choosing fitness function for multi-objective optimization.
    :return:

    '''
    # ratio of first peak duration. scaled by geometric mean value of parameter as t~1/V, we want to avoid optimized to small parameter.
    scaled_first_peak_time_duration_ratio = first_peak_Time_duration_ratio  * parameter_geometric_mean_ratio

    max_energy_fitness_contribution = 0.5 * pow(max_energy_change, 2)

    localization_duration_ratio_contribution = 10 * pow(Localization_duration_ratio , 2)

    first_peak_duration_contribution = scaled_first_peak_time_duration_ratio * 20

    fitness_func_value = max_energy_fitness_contribution + localization_duration_ratio_contribution + first_peak_duration_contribution

    return fitness_func_value , max_energy_fitness_contribution, localization_duration_ratio_contribution , first_peak_duration_contribution
