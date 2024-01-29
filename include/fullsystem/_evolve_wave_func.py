import numpy as np
from include.detector.__init__ import Detector
from include.constructing_state_module import binary_search_qn_list

import numba
from numba import jit

'''
Evolve Hamiltonian on basis set and compute energy of photon & detectors.
'''

@jit(nopython = True)
def wave_func_sum(array_sum, part_add, row_index):
    '''
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

# ----------- initialize wave function -----------------------------
def initialize_wave_function(self):
    '''
    initialize full system's initial wave function.
    :param self: class pointer
    :return:
    '''
    # initialize wave_func for d1, d2
    # assume initially detector1 and detector2 are in pure state.
    # position1 and position2 are positions of initial detector state in detector wave function.
    self.detector1.initialize_wave_function()
    self.detector2.initialize_wave_function()

    detector1_initial_state_index = self.detector1.output_detector_initial_state_index()
    detecto2_initial_state_index = self.detector2.output_detector_initial_state_index()

    self._wave_function = np.zeros(self._basis_set_state_num, dtype=np.complex)

    for i in range(self._basis_set_state_num):
        if self.dstate1[i] == detector1_initial_state_index and self.dstate2[i] == detecto2_initial_state_index:
            if self.pstate[i] == 1:
                self._wave_function[i] = self.initial_photon_wave_function[0]
            if self.pstate[i] == 2:
                self._wave_function[i] = self.initial_photon_wave_function[1]

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

def evolve_wave_function(self):
    '''
    evolve wave function forward in time.
    compute photon_energy, energy in detector1 and energy in detector2.
    :param self: class pointer for full system.
    :return:
    '''
    final_time = self.time_duration
    output_time_step = self.output_time_step

    # define time step to do simulation
    max_h_element = np.max(np.abs(self._mat))
    time_step = 0.02 / max_h_element

    # output step number and total_step_number
    output_step_number = max(int(output_time_step / time_step), 1)
    total_step_number = int(final_time / time_step)

    real_part_wave_func = np.real(self._wave_function)
    imag_part_wave_func = np.imag(self._wave_function)

    # transform mat, irow, icol to numpy matrix to facilitate further manipulation.
    self.numpy_array_for_data()
    self.photon_hamiltonian_in_full_basis_set.numpy_array_for_data()
    self.detector1_hamiltonian_in_full_basis_set.numpy_array_for_data()
    self.detector2_hamiltonian_in_full_basis_set.numpy_array_for_data()

    detector1_energy_list = []
    detector2_energy_list = []
    photon_energy_list = []

    t = 0
    time_list = []
    wave_function_list = []

    for step in range(total_step_number):
        # evaluate result. output photon_energy, detector1_energy, detector2_energy
        if step % output_step_number == 0:
            self._wave_function = real_part_wave_func + 1j * imag_part_wave_func

            wave_function_list.append(self.wave_function)

            photon_energy = self._evaluate_photon_energy()

            d1_energy = self._evaluate_detector_energy(self.detector1_hamiltonian_in_full_basis_set)

            d2_energy = self._evaluate_detector_energy(self.detector2_hamiltonian_in_full_basis_set)

            photon_energy_list.append(photon_energy)
            detector1_energy_list.append(d1_energy)
            detector2_energy_list.append(d2_energy)

            if  abs(photon_energy - self.init_photon_energy) > 0.1:
                raise NameError("Error for photon energy convergence")

            time_list.append(t)

        # evolve wave function. simple SUR algorithm: https://doi.org/10.1016/0009-2614(94)01474-A
        # real_part = real_part + H * dt * imag_part
        real_part_change = self._mat_array * imag_part_wave_func[self._icol_array] * time_step
        # use numba to speed up H = H + H_change. For numba, see : https://numba.pydata.org/
        real_part_wave_func = wave_func_sum(real_part_wave_func, real_part_change, self._irow_array)

        # imag_part = imag_part - H * dt * real_part
        imag_part_change = - self._mat_array * real_part_wave_func[self._icol_array] * time_step
        # use numba to speed up
        imag_part_wave_func = wave_func_sum(imag_part_wave_func, imag_part_change, self._irow_array)

        t = t + time_step

    detector1_energy_list = np.array(detector1_energy_list)
    detector2_energy_list = np.array(detector2_energy_list)
    photon_energy_list = np.array(photon_energy_list)
    time_list = np.array(time_list)

    # check energy conservation
    check_energy_conservation(time_step, time_list, detector1_energy_list, detector2_energy_list, photon_energy_list)

    return photon_energy_list, detector1_energy_list, detector2_energy_list, time_list

# ---------- evaluate photon , d1, d2 energy --------------
def _evaluate_photon_energy(self):
    '''
    evaluate the energy of photon
    :param self: class pointer
    :return:
    '''
    # compute H * phi, here H is photon hamiltonian on full system's basis set.
    # H[irow, icol], here h_phi[i] is the value of (H * phi) [irow[i]]
    h_phi = self.photon_hamiltonian_in_full_basis_set.get_mat_array() * self._wave_function[self._icol_array]

    # compute H * phi
    h_phi_wave_function = np.zeros(self._basis_set_state_num, dtype=np.complex)
    h_phi_wave_function = wave_func_sum(h_phi_wave_function, h_phi, self.photon_hamiltonian_in_full_basis_set.get_irow_array())
    # compute <phi | H | phi>. photon energy.
    photon_energy = np.sum(np.real(np.conjugate(self._wave_function) * h_phi_wave_function))

    return photon_energy

def _evaluate_detector_energy(self, detector_hamiltonian):
    '''
    compute energy of detector.
    :param self: class pointer
    :param detector_hamiltonian: detector hamiltonian class pointer.
    :return:
    '''
    # H_{detector} * |\psi>
    # H[irow, icol], here h_phi[i] is the value of (H * phi) [irow[i]]
    h_phi = detector_hamiltonian.get_mat_array() * self._wave_function[detector_hamiltonian.get_icol_array()]

    # compute H * phi
    h_phi_wave_function = np.zeros(self._basis_set_state_num, dtype=np.complex)
    h_phi_wave_function = wave_func_sum(h_phi_wave_function, h_phi, detector_hamiltonian.get_irow_array())

    # compute <phi | H | phi>. detector energy.
    detector_energy = np.sum(np.real(np.conjugate(self._wave_function) * h_phi_wave_function))

    return detector_energy


