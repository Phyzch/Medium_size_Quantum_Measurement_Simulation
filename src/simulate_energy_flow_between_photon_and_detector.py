import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()
import include.fullsystem.__init__

def simulate_full_system_quantum_dynamics(full_system_instance, off_diagonal_coupling):
    '''
    evolve the fullsystem instance's wave function.
    compute the change of detector energy and photon energy.
    :param full_system_instance: instance of full_system class
    :param off_diagonal_coupling: off-diagonal coupling terms in full_system Hamiltonian.
    :return:
    '''
    # construct off diagonal part for full system Hamiltonian
    full_system_instance.construct_full_system_Hamiltonian_offdiagonal_part(off_diagonal_coupling)

    # initialize wave function
    # this function is in /fullsystem/_evolve_wave_func.py.
    full_system_instance.initialize_wave_function()

    # Evolve dynamics of full system
    photon_energy_list, detector1_energy_list, detector2_energy_list, time_list = full_system_instance.evolve_wave_function()

    # Analyze_simulation result of full_system. (wave function)
    init_photon_energy = full_system_instance.init_photon_energy

    photon_energy_list = photon_energy_list / init_photon_energy

    detector1_energy_list_change = (detector1_energy_list - detector1_energy_list[0]) / init_photon_energy
    detector2_energy_list_change = (detector2_energy_list - detector2_energy_list[0]) / init_photon_energy

    return photon_energy_list, detector1_energy_list_change, detector2_energy_list_change, time_list


def extract_first_peak_from_all_peaks(peak_index_list):
    '''
    extract list of consecutive number from list and set this as first peak.
    :param peak_index_list: list of index for the detector energy's peak
    :return:
    '''
    list_length = len(peak_index_list)
    first_peak_list = []

    if list_length > 1:
        for i in range(1, list_length):
            first_peak_list.append(peak_index_list[i - 1])
            if peak_index_list[i] - peak_index_list[i - 1] != 1:
                break

    else:
        if list_length == 1:
            first_peak_list = [peak_index_list[0]]

    return first_peak_list

def analyze_peak_of_detector_energy_submodule(time_list, detector_energy_change, detector_energy_change_peak_index):
    '''
    Three variables are of particular interest:
    1. time duration for first peak
    2. maximum energy change for first peak
    3. time for localization.
    :param: time_list: list of time
    :param: detector_energy_change: change of detector energy.
    :param: detector_energy_change_peak_index: index for the peak of detector's energy change
    :return : first_peak_time_duration, first_peak_maximum , localization_time_duration_ratio
    '''
    time_step = time_list[1] - time_list[0]
    final_time = time_list[-1]

    # time range for localization.
    localization_time_duration = len(detector_energy_change_peak_index) * time_step
    localization_time_duration_ratio = localization_time_duration / final_time

    # first peak's index and peak value:  (here peak is peak for energy change)
    index_for_detector_energy_first_peak = extract_first_peak_from_all_peaks(detector_energy_change_peak_index)

    first_peak_time = np.array(time_list)[index_for_detector_energy_first_peak]

    length_of_first_peak_index_list = len(index_for_detector_energy_first_peak)

    energy_of_first_peak = detector_energy_change[index_for_detector_energy_first_peak]

    # time_duration of the first peak
    first_peak_time_duration = (first_peak_time[-1] - first_peak_time[0]) if length_of_first_peak_index_list > 0 else 0
    first_peak_time_duration_ratio = first_peak_time_duration / final_time

    # the maximum energy change for the first peak.
    # We compute the maximum energy for first peak.
    first_peak_maximum = max(energy_of_first_peak) if length_of_first_peak_index_list != 0 else 0


    return first_peak_time_duration_ratio , first_peak_maximum , localization_time_duration_ratio

def decide_side_of_detector_energy_localization(highest_peak_bool,
                                                max_left_detector_energy_change, max_right_detector_energy_change,
                                                left_detector_energy_peak_index, right_detector_energy_peak_index):
    '''
    decide the side of localization for the full system using two detectors' energy.
    if we use the criterion of the highest peak energy (the side with the highest photon energy is decided as the side
    of localization, then highest_peak_bool = True.
    If highest_peak_bool = False, we choose the side of the localization according to the criteria:
    which side show the earliest energy peak.
    :param highest_peak_bool: bool of the highest energy
    :param max_left_detector_energy_change: maximum energy change in the left detector.
    :param max_right_detector_energy_change:  maximum energy change in the right detector.
    :param left_detector_energy_peak_index:  the index of the peak of left detector's energy change.
    :param right_detector_energy_peak_index: the index of the peak of right detector's energy change.
    :return: localization_side.
    localization_side = "left": localize to left hand side. localization_side = "right": localize to right hand side.

    '''
    localization_side = ""
    if highest_peak_bool:
        # ------- criteria: use peak's highest energy. ---------------------
        if max_left_detector_energy_change > max_right_detector_energy_change :
            localization_side = "left"
        else:
            localization_side = "right"
    else:
        # --------- criteria : use earliest peak as criteria ------------
        localization_bool = 0
        if len(left_detector_energy_peak_index) == 0:
            # first peak appear at right.
            localization_side = "right"
        elif len(right_detector_energy_peak_index) == 0:
            # first peak appears at left.
            localization_side = "left"
        else:
            if  left_detector_energy_peak_index[0] < right_detector_energy_peak_index[0]:
                # first peak appears at left
                localization_side = "left"

            elif left_detector_energy_peak_index[0] > right_detector_energy_peak_index[0]:
                # first peak appears at right
                localization_side = "right"

    return localization_side

def analyze_peak_and_peak_duration(left_detector_energy_change, right_detector_energy_change, time_list, highest_peak_bool) :
    '''

    :param left_detector_energy_change: change of energy for l.h.s.
    :param right_detector_energy_change: change of energy for r.h.s.
    :param time_list: list of time.
    :param highest_peak_bool : if bool == True, we choose highest peak as criteria for localization.
                               if bool == False, we choose first peak as criteria for localization.
    :return: first_peak_time_duration_ratio: ratio of the time for the first peak over the whole time duration.
             first_peak_maximum: maximum value of the first peak
             localization_time_duration_ratio: ratio of the localization time over the whole time duration.
             localization_side: side of localization. if localization_side = "left": localize to the left.
                                                      if localization_side = "right": localize to the right.
    '''

    max_lhs_energy_change = max(left_detector_energy_change)
    max_rhs_change = max(right_detector_energy_change)
    max_energy_change = max(max_lhs_energy_change, max_rhs_change)

    # if we use first peak instead of highest peak as criteria. (highest_peak_bool == False)
    # We first find highest peak.
    # Then set ratio * highest peak as threshold to be treated as a peak. Then we find earliest peak in simulation.
    peak_criteria_ratio = 0.7
    criteria_for_peak = peak_criteria_ratio * max_energy_change


    left_detector_energy_peak_index = [i for i in range(len(left_detector_energy_change))
                                       if left_detector_energy_change[i] > criteria_for_peak]
    right_detector_energy_peak_index = [i for i in range(len(right_detector_energy_change))
                                        if right_detector_energy_change[i] > criteria_for_peak]

    # ----  decide which side this simulation localize ---------
    localization_side = decide_side_of_detector_energy_localization(highest_peak_bool, max_lhs_energy_change, max_rhs_change,
                                                                    left_detector_energy_peak_index, right_detector_energy_peak_index)


    if localization_side == "left":
        # localize to the left
        first_peak_time_duration_ratio, first_peak_maximum , localization_time_duration_ratio = (
            analyze_peak_of_detector_energy_submodule(time_list, left_detector_energy_change, left_detector_energy_peak_index))
    else:
        # localization to the right.
        first_peak_time_duration_ratio, first_peak_maximum , localization_time_duration_ratio = (
            analyze_peak_of_detector_energy_submodule(time_list, right_detector_energy_change, right_detector_energy_peak_index))

    return first_peak_time_duration_ratio, first_peak_maximum, localization_time_duration_ratio, localization_side



def compute_normalized_offdiagonal_coupling_geometric_mean(off_diagonal_coupling, parameter_number, parameter_range):
    '''
    compute normalized geometric mean of off-diagonal coupling
    :param off_diagonal_coupling: off-diagonal coupling in full system's hamiltonian.
    This is the parameter to be optimized.
    :param parameter_number: number of parameters to be optimized.
    :param parameter_range: the range of off-diagonal coupling parameter.
    :return:
    '''
    # pow(  \prod coupling , 1/N) / parameter_range
    # coupling ^{1/N}
    off_diagonal_coupling_pow = np.power(off_diagonal_coupling , 1 / parameter_number)
    off_diagonal_coupling_geometric_mean = np.product(off_diagonal_coupling_pow)

    normalized_geometric_mean = off_diagonal_coupling_geometric_mean / parameter_range

    return normalized_geometric_mean

def compute_fitness_function_submodule(first_peak_time_duration_ratio, first_peak_maximum,
                                       localization_time_duration_ratio, parameter_geometric_mean_ratio):
    '''
    We need to be very careful when defining fitness function.

    localization_duration_ratio is in range[0,1], max_energy_change is in range[0,1],  first_peak_time_duration_ratio is also in range[0,1]

    parameter_geometric_mean_ratio is geometric mean of parameter / upper limit you set  for parameter.
    This is to make sure we don't sample extremely small parameter to get long localization time.

    This choice of fitness function is tricky and ad-hoc, try different choices to see which works best.
    See p.97 in Practical Genetic algorithms (R.L.Haupt & S.E.Haupt) about choosing fitness function for multi-objective optimization.
    :param: first_peak_time_duration_ratio: ratio of the first peak duration over the whole time.
    :param: first_peak_maximum: maximum of the first peak of energy.
    :param: localization_time_duration_ratio: ratio of the localization time over the whole time duration.
    :param: parameter_geometric_mean_ratio: geometric mean of the parameter / parameter_range.
    :return:
    The fitness function is the sum of different parts. We return individual part the contribute to the fitness and the overall fitness function itself.
    fitness_func_value : value of the overall fitness function.
    peak_height_contribution: contribution from the height of the energy peak to the fitness function.
    localization_time_contribution: contribution from the localization time to the fitness function.
    first_peak_time_contribution: contribution from the first peak's time to the fitness function.
    '''
    peak_height_contribution = 0.5 * pow(first_peak_maximum, 2)

    localization_time_contribution = 10 * pow(localization_time_duration_ratio, 2)

    # ratio of first peak duration. scaled by geometric mean value of parameter as t~1/V, we want to avoid optimized to small parameter.
    scaled_first_peak_time_duration_ratio = first_peak_time_duration_ratio * parameter_geometric_mean_ratio
    first_peak_time_contribution = scaled_first_peak_time_duration_ratio * 20

    fitness_func_value = peak_height_contribution + localization_time_contribution + first_peak_time_contribution

    return fitness_func_value , peak_height_contribution, localization_time_contribution , first_peak_time_contribution

def fitness_function_for_individual_full_system_instance(off_diagonal_coupling, seed_data):
    '''
    function to compute the fitness function for individual.
    :param off_diagonal_coupling: value of off-diagonal coupling for the FullSystem instance.
    :param seed_data: seed data for genetic algorithm: [coupling_parameter_range, full_system_instance, parameter_number, highest_peak_bool]
    :return:
    '''
    # data is maximum range of coupling strength
    coupling_parameter_range , full_system_instance, coupling_parameter_number , highest_peak_bool  = seed_data

    # construct off diagonal part of Hamiltonian. and initialize wave function
    (photon_energy_list, detector1_energy_list_change,
     detector2_energy_list_change, time_list) = simulate_full_system_quantum_dynamics(full_system_instance, off_diagonal_coupling)

    # decide the side of the localization and duration for the localization of the detector energy to one side.
    first_peak_time_duration_ratio, first_peak_maximum, localization_time_duration_ratio, localization_side = \
     analyze_peak_and_peak_duration(detector1_energy_list_change, detector2_energy_list_change, time_list, highest_peak_bool= highest_peak_bool)

    # compute the ratio of the geometric mean of the off-diagonal coupling compared to coupling parameter range.
    normalized_geometric_mean = compute_normalized_offdiagonal_coupling_geometric_mean(off_diagonal_coupling,
                                                                                       coupling_parameter_number, coupling_parameter_range)

    # compute the fitness function using the detector energy and photon energy.
    (fitness_func_value , peak_height_fitness_contribution,
     localization_time_contribution , first_peak_time_contribution)  \
        = compute_fitness_function_submodule(first_peak_time_duration_ratio, first_peak_maximum,
                                             localization_time_duration_ratio, normalized_geometric_mean)

    return fitness_func_value


