import numpy as np
from include.detector.__init__ import Detector
from include.constructing_state_module import binary_search_qn_list

import numba
from numba import jit

'''
Evolve Hamiltonian on basis set and compute energy of photon & detectors.
'''

@jit(nopython = True)
def wave_func_sum(original_value, part_add, index):
    '''
    original_value [ index[i] ] = original_value [ index[i] ] + part_add[i]
    :return: original_value
    '''
    Len = np.shape(index)[0]
    for i in range(Len):
        index_i = index[i]
        original_value[index_i] = original_value[index_i] + part_add[i]

    return original_value

# ----------- initialize wave function -----------------------------
def initialize_wave_function(self):
    # initialize wave_func for d1, d2
    # assume initially d1, d2 in pure state.
    # position1 and position2 is postion of initial detector state in detector wave function.
    self.detector1.initialize_wave_function()
    self.detector2.initialize_wave_function()

    position1 = self.detector1.output_detector_initial_state_index()
    position2 = self.detector2.output_detector_initial_state_index()

    self._wave_function = np.zeros(self._basis_set_state_num, dtype=np.complex)

    for i in range(self._basis_set_state_num):
        if self.dstate1[i] == position1 and self.dstate2[i] == position2:
            if self.pstate[i] == 1:
                self._wave_function[i] = self.initial_photon_wave_function[0]
            if self.pstate[i] == 2:
                self._wave_function[i] = self.initial_photon_wave_function[1]

# ----------- Evolve Schrodinger equation on basis set. -------------------------
def check_energy_conservation(time_step, Time_list, d1_energy_list, d2_energy_list, photon_energy_list ):
    Total_energy = d1_energy_list + d2_energy_list + photon_energy_list
    total_energy_len = len(Total_energy)
    for i in range(total_energy_len):
        if( abs(Total_energy[i] - 1) > 0.1 ):
            print("simulation time step:  " + str(time_step))
            print('time: ' + str(Time_list[i]))
            print('photon energy :  ' + str(photon_energy_list[i]) + ' detector1 energy:  ' + str(d1_energy_list[i]) +"   detector2 energy:  " + str(d2_energy_list[i]) )
            raise NameError("SUR algorithm do not converge energy. Check code for error")

def Evolve_dynamics(self):
    final_time = self.time_duration
    output_time_step = self.output_time_step

    # define time step to do simulation
    max_H_element = np.max(np.abs(self.full_H._mat))
    time_step = 0.02 / (max_H_element)

    # output step number and total_step_number
    output_step_number = max(int(output_time_step / time_step), 1)
    total_step_number = int(final_time / time_step)

    real_part = np.real(self._wave_function)
    imag_part = np.imag(self._wave_function)

    # transform mat, irow, icol to numpy matrix to facilitate further manipulation.
    self.full_H.numpy_array_for_data()
    self.photon_H.numpy_array_for_data()
    self.d1_H.numpy_array_for_data()
    self.d2_H.numpy_array_for_data()

    d1_energy_list = []
    d2_energy_list = []
    photon_energy_list = []

    t = 0
    Time_list = []
    wave_function_list = []

    for step in range(total_step_number):
        # evaluate result. output photon_energy, detector1_energy, detector2_energy
        if (step % output_step_number == 0):
            self._wave_function = np.array([np.complex(real_part[i], imag_part[i]) for i in range(self._basis_set_state_num)])

            # wave_function_list.append(self.wave_function)
            photon_energy = self._evaluate_photon_energy()
            d1_energy = self._evaluate_d_energy(self.d1_H)
            d2_energy = self._evaluate_d_energy(self.d2_H)

            photon_energy_list.append(photon_energy)
            d1_energy_list.append(d1_energy)
            d2_energy_list.append(d2_energy)

            if (step == 0 and abs(photon_energy - self.init_photon_energy) > 0.1):
                raise NameError("Error for photon energy convergence")

            Time_list.append(t)

        # Evolve wave function. simple SUR algorithm: https://doi.org/10.1016/0009-2614(94)01474-A
        # real_part = real_part + H * dt * imag_part
        real_part_change = self.full_H._mat_array * imag_part[self.full_H._icol_array] * time_step
        # use numba to speed up H = H + H_change. For numba, see : https://numba.pydata.org/
        real_part = wave_func_sum(real_part, real_part_change, self.full_H._irow_array)

        # imag_part = imag_part - H * dt * real_part
        imag_part_change = -self.full_H._mat_array * real_part[self.full_H._icol_array] * time_step
        # use numba to speed up
        imag_part = wave_func_sum(imag_part, imag_part_change, self.full_H._irow_array)

        t = t + time_step

    d1_energy_list = np.array(d1_energy_list)
    d2_energy_list = np.array(d2_energy_list)
    photon_energy_list = np.array(photon_energy_list)
    Time_list = np.array(Time_list)

    # check energy conservation
    # check_energy_conservation(time_step, Time_list, d1_energy_list, d2_energy_list, photon_energy_list)

    return photon_energy_list, d1_energy_list, d2_energy_list, Time_list

# ---------- evaluate photon , d1, d2 energy --------------
def _evaluate_photon_energy(self):
    # use self.mat_photon and self.photon_H.irow. self.photon_H.icol
    H_phi = self.photon_H._mat_array * self._wave_function[self.photon_H._icol_array]

    H_phi_wave_function = np.zeros(self._basis_set_state_num, dtype=np.complex)
    H_phi_wave_function = wave_func_sum(H_phi_wave_function, H_phi, self.photon_H._irow_array)

    photon_energy = np.sum(np.real(np.conjugate(self._wave_function) * H_phi_wave_function))

    return photon_energy

def _evaluate_d_energy(self, d_H):
    # H * |\psi>
    H_phi = d_H._mat_array * self._wave_function[d_H._icol_array]

    H_phi_wave_function = np.zeros(self._basis_set_state_num, dtype=np.complex)
    H_phi_wave_function = wave_func_sum(H_phi_wave_function, H_phi, d_H._irow_array)

    d_energy = np.sum(np.real(np.conjugate(self._wave_function) * H_phi_wave_function))

    return d_energy


