import os

import numpy as np
from matplotlib import pyplot as plt, gridspec as gridspec

import simulate_energy_flow_between_photon_and_detector


from include.util import rank


def save_simulation_result(*args):
    best_param, best_fitness, parameter_number, parameter_range, time_list, \
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
        normalized_geometric_mean = simulate_energy_flow_between_photon_and_detector.compute_normalized_offdiagonal_coupling_geometric_mean(best_param, parameter_number, parameter_range)

        f1.write('geometric mean : \n')
        f1.write(str(normalized_geometric_mean) + " \n")

        f1.write('best fitness ' + "\n")
        f1.write(str(best_fitness) + '\n')

        # fit function
        best_fitness_by_fitting, max_energy_fitness_contribution, localization_duration_ratio_contribution, first_peak_duration_contribution = simulate_energy_flow_between_photon_and_detector.compute_fitness_function_submodule(
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
        write_data(f1, time_list, 'Time: ')
        write_data(f1, d1_energy_list_change, "el:  ")
        write_data(f1, d2_energy_list_change, "er:  ")
        write_data(f1, photon_energy_list, "e_photon:  ")


def output_full_system_state_and_coupling_info(full_system_instance):
    if rank == 0:
        # print information about structure of system
        full_system_instance.output_state_qn_number_list()
        full_system_instance.detector1.output_detector_anharmonic_coupling_state_pairs()
        full_system_instance.detector2.output_detector_anharmonic_coupling_state_pairs()

        full_system_instance.output_off_diagonal_coupling_state_pairs_info()
        print("parameter number for detector1: " + str(full_system_instance.detector1.show_offdiag_matrix_num()))
        print("parameter number for detector2: " + str(full_system_instance.detector2.show_offdiag_matrix_num()))
        print("parameter number for couplings betweeen detector and system:  " + str(full_system_instance.offdiag_param_num
                                                                                   - full_system_instance.detector1.show_offdiag_matrix_num()
                                                                                   - full_system_instance.detector2.show_offdiag_matrix_num()
                                                                                   )
              )


def plot_simulation_result(*args):
    photon_energy_list, d1_energy_list_change, d2_energy_list_change, time_list, file_path = args
    # configure fig, ax
    fig1 = plt.figure(figsize=(15, 15))
    spec = gridspec.GridSpec(nrows=1, ncols=1, figure=fig1)
    spec.update(hspace=0.5, wspace=0.3)
    ax1 = fig1.add_subplot(spec[0, 0])
    ax1.plot(time_list, d1_energy_list_change, label='left photon localization')
    ax1.plot(time_list, d2_energy_list_change, label='right photon localization')
    ax1.plot(time_list, photon_energy_list, label='photon energy')

    ax1.set_xlabel('time')
    ax1.set_ylabel('E')

    ax1.legend(loc='best')

    # save figure.
    fig_name = "best_simulation_result.svg"
    fig_name = os.path.join(file_path, fig_name)

    fig1.savefig(fig_name)


def write_data(f, data_list, symbol):
    f.write(symbol + "\n")
    for data in data_list:
        f.write(str(round(data, 4)) + " ")
    f.write("\n")
