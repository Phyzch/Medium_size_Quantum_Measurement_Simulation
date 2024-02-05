import numpy as np
from numba import jit

'''
Evolve Hamiltonian on basis set and compute energy of photon & detectors.
'''

@jit(nopython = True)
def wave_func_sum(array_sum, part_add, row_index):
    '''
    Here we need to use numba to pre-compile the code. Otherwise, doing this calculation use CPython will be very slow.
    This is because I can not vectorize this part of the code.
    original_value [ index[i] ] = original_value [ index[i] ] + part_add[i]
    :param: array_value: value of the array.
    :param: part_add: the part we need to add
    :param: array_index : index of the array
    :return: array_value
    '''
    length = np.shape(row_index)[0]
    for i in range(length):
        row_index_i = row_index[i]
        array_sum[row_index_i] = array_sum[row_index_i] + part_add[i]

    return array_sum


# ----------- Evolve Schrodinger equation on basis set. -------------------------
def check_energy_conservation(time_step, time_list, d1_energy_list, d2_energy_list, photon_energy_list):
    '''
    check whether the total energy is conserved or not.
    :param time_step:
    :param time_list:
    :param d1_energy_list:
    :param d2_energy_list:
    :param photon_energy_list:
    :return:
    '''
    total_energy = d1_energy_list + d2_energy_list + photon_energy_list
    total_energy_length = len(total_energy)
    for i in range(total_energy_length):
        if abs(total_energy[i] - 1) > 0.1 :
            print("simulation time step:  " + str(time_step))
            print('time: ' + str(time_list[i]))
            print('photon energy :  ' + str(photon_energy_list[i]) + ' detector1 energy:  ' + str(d1_energy_list[i]) +"   detector2 energy:  " + str(d2_energy_list[i]) )
            raise NameError("SUR algorithm do not converge energy. Check code for error")





