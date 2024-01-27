import numpy as np
from include.detector.__init__ import Detector
from include.constructing_state_module import binary_search_qn_list

'''
first part of constructing Hamiltonian. This only have to be called once.
We do following : 1. compute energy of full_system  : self.__compute_initial_energy()
                  2. let detector construct Hamiltonian and coupling info: self.detector1/2.construct_detector_Hamiltonian_part1()
                  3. construct diagonal part of Hamiltonian and state for full_system : self.__construct_full_system_diagonal_Hamiltonian()
                  4. construct irow, icol for off-diagonal coupling between state : self.__construct_offdiag_dd_pd_coup() , self.__construct_intra_detector_coupling()
'''

def construct_full_system_hamiltonian_part1(self):
    self._compute_initial_energy()

    self.detector1.construct_detector_Hamiltonian_part1()
    self.detector2.construct_detector_Hamiltonian_part1()

    # construct basis set for full system.
    # also construct diagonal part of Hamiltonian
    self._construct_full_system_basis_set()

    # shift Hamiltonian's diag part by initial energy to speed up propagation of wave function.
    self._shift_Hamiltonian()

    self.offdiag_param_num = self.offdiag_param_num + self.detector1.show_offdiag_matrix_num()
    self.offdiag_param_num = self.offdiag_param_num + self.detector2.show_offdiag_matrix_num()

    # compute the parameter number and irow, icol for coupling between two detectors
    self._construct_offdiag_detector_detector_coupling()

    # compute the parameter number and irow, icol for coupling between photon and detector
    self._construct_offdiag_photon_detector_coupling()

    # compute position of intra-detector coupling
    self._construct_intra_detector_coupling()


def _compute_initial_energy(self):
    self.initial_energy = 0
    d1_energy = np.sum(np.array(self.detector1.show_mode_frequency()) * np.array(self.detector1.show_initial_state_qn()))
    d2_energy = np.sum(np.array(self.detector2.show_mode_frequency()) * np.array(self.detector2.show_initial_state_qn()))

    self.initial_energy = self.init_photon_energy + d1_energy + d2_energy


def _construct_full_system_basis_set(self):
    '''
    construct state and diagonal part of Hamiltonian.
    impose energy window : states included should satisfy : |E - E_init | <= energy_window
    :return:
    '''
    basis_set_state_index  = 0

    for i in range(self.photon_state_num):
        for j in range(self.detector1.show_state_num()):
            for k in range(self.detector2.show_state_num()):
                energy = self.photon_state_energy[i] + self.detector1.get_basis_set_state_energy(j) + \
                         self.detector2.get_basis_set_state_energy(k)

                if abs(energy - self.initial_energy) <= self.energy_window_for_basis_set_state:
                    # record photon state and detector state index for state in full system.
                    self.pstate.append(i)
                    self.dstate1.append(j)
                    self.dstate2.append(k)

                    state_quantum_number = [self.photon_state_qn[i], self.detector1.get_basis_set_state_quantum_number(j).tolist(),
                                  self.detector2.get_basis_set_state_quantum_number(k).tolist()]

                    # add basis set state qn and energy to list.
                    self._full_system_add_basis_set_state(state_quantum_number, energy, basis_set_state_index)

                    # Hamiltonian for photon, d1, d2
                    self.photon_hamiltonian_in_full_basis_set.append_matrix_element(self.photon_state_energy[i],
                                                                                    basis_set_state_index, basis_set_state_index)
                    self.detector1_hamiltonian_in_full_basis_set.append_matrix_element(self.detector1.get_basis_set_state_energy(j),
                                                                                       basis_set_state_index, basis_set_state_index)
                    self.detector2_hamiltonian_in_full_basis_set.append(self.detector2.get_basis_set_state_energy(k),
                                                                        basis_set_state_index, basis_set_state_index)

                    basis_set_state_index = basis_set_state_index + 1

    self._basis_set_state_num = basis_set_state_index


    # update diagonal Hamiltonian and Hamiltonian itself.
    self._add_hamiltonian_diagonal_part()




def _shift_Hamiltonian(self):
    '''
    shift Hamiltonian by energy : <\psi | H | \psi>
    '''
    for i in range(self._basis_set_state_num):
        self._mat[i] = self._mat[i] - self.initial_energy

def _construct_intra_detector_coupling_submodule(self, i, j, di, dj, detector, detector_hamiltonian_in_full_basis_set,
                                                 intra_detector_coupling_index):
    '''
    :param i, j: state index in full_system
    :param di, dj : state index in detector .
    :return:
    '''
    coupling_value = - 1.0     # this coupling value will be substituted later.
    detector_state_num = detector.show_state_num()
    detector_mat_num = detector.show_mat_num()

    for k in range(detector_state_num, detector_mat_num):
        # k is off-diagonal matrix index in detector's Hamiltonian
        if detector.get_irow(k) == di and detector.get_icol(k) == dj:
            # add intra-detector anharmonic coupling to detector hamiltonian in full basis set.
            detector_hamiltonian_in_full_basis_set.append_matrix_element(coupling_value, i, j)
            detector_hamiltonian_in_full_basis_set.append_matrix_element(coupling_value, j, i)

            # index for anharmonic coupling in detector hamiltonian is recorded for reference.
            intra_detector_coupling_index.append(k)

            # for evolve wave function, we need to record off-diagonal matrix element symmetrically.
            self.append_matrix_element(coupling_value, i, j)
            self.append_matrix_element(coupling_value, j, i)

            break

def _construct_intra_detector_coupling(self):
    # first include anharmonic coupling in detector 1.
    for i in range(self._basis_set_state_num):
        for j in range(i + 1, self._basis_set_state_num):
            ss = self.pstate[i] - self.pstate[j]

            # i , j stands for different state.  1, 2 stand for detector 1 and detector 2
            di1 = self.dstate1[i]
            di2 = self.dstate2[i]
            dj1 = self.dstate1[j]
            dj2 = self.dstate2[j]

            # coupling in detector 1
            if ss == 0 and di1 != dj1 and di2 == dj2:
                self._construct_intra_detector_coupling_submodule(i,j, di1, di2, self.detector1,
                                                                  self.detector1_hamiltonian_in_full_basis_set,
                                                                  self.d1_coupling_index)

    # then include anharmonic coupling in detector 2.
    for i in range(self._basis_set_state_num):
        for j in range(i + 1, self._basis_set_state_num):
            ss = self.pstate[i] - self.pstate[j]

            # i , j stands for different state.  1, 2 stand for detector 1 and detector 2
            di1 = self.dstate1[i]
            di2 = self.dstate2[i]
            dj1 = self.dstate1[j]
            dj2 = self.dstate2[j]

            # coupling in detector2
            if ss == 0 and di1 == dj1 and di2 != dj2:
                self._construct_intra_detector_coupling_submodule(i, j, di2, dj2, self.detector2,
                                                                  self.detector2_hamiltonian_in_full_basis_set,
                                                                  self.d2_coupling_index)


def examine_dd_coupling( state_quantum_number_di, state_quantum_number_dj):
    '''
    check whether the quantum number of the detector for the state di and the state dj only differ by one.
    :param: state_quantum_number_di : quantum number of detector states dj
    :param: state_quantum_number_dj : quantum number of detectors states di
    :return:
    '''
    quantum_number_difference_sum = np.sum( np.abs(state_quantum_number_di - state_quantum_number_dj) )
    if quantum_number_difference_sum == 1:
        succeed = True
    else:
        succeed = False
    return succeed

def _include_detector_detector_coupling(self, di1, dj1, di2, dj2, i, j):
    '''
    :param: di1: basis set state index for detector state in detector 1 for full system state i.
    :param dj1: basis set state index for detector state in detector 1 for full system state j
    :param di2: basis set state index for detector state in detector 2 for full system state i.
    :param dj2: basis set state index for detector state in detector 2 for full system state j
    :param i: state index in full system basis set
    :param j: state index in full system basis set
    include coupling between detectors
    :return:
    '''
    coupling_value = - 1.0     # this coupling value will be substituted later.
    d1_succeed = examine_dd_coupling(self.detector1.get_basis_set_state_quantum_number(di1),
                                     self.detector1.get_basis_set_state_quantum_number(dj1))
    d2_succeed = examine_dd_coupling(self.detector2.get_basis_set_state_quantum_number(di2),
                                     self.detector2.get_basis_set_state_quantum_number(dj2))

    if d1_succeed and d2_succeed:
        self.dd_coupling_irow.append(i)
        self.dd_coupling_icol.append(j)

        self.append_matrix_element(coupling_value, i, j)
        # lower triangular part.
        self.append_matrix_element(coupling_value, j, i)

        self.dd_coupling_num = self.dd_coupling_num + 1
        self.offdiag_param_num = self.offdiag_param_num + 1

def _construct_offdiag_detector_detector_coupling(self):
    '''
    construct off diagonal coupling between two detectors
    :param self: class pointer
    :return:
    '''

    for i in range(self._basis_set_state_num):
        for j in range(i+1, self._basis_set_state_num):
            ss = self.pstate[i] - self.pstate[j]

            di1 = self.dstate1[i]
            di2 = self.dstate2[i]

            dj1 = self.dstate1[j]
            dj2 = self.dstate2[j]

            # photon state index : 0 <-> [0,0], 1 <-> [1,0], 2 <-> [0,1],
            # no coupling between states with photon state [1,0] and [0,1]
            if self.pstate[i] + self.pstate[j] == 3:
                continue

            # coupling between detector 1 and detector 2
            if ss == 0 and di1 != dj1 and di2 != dj2:
                self._include_detector_detector_coupling(di1, dj1, di2, dj2, i, j)

def _include_photon_detector_coupling(self, state_quantum_number_di, state_quantum_number_dj, i, j):
    '''
    :param: di: basis set state index for detector state in detector (1 or 2) for full system state i.
    :param dj: basis set state index for detector state in detector (1 or 2) for full system state j
    :param i: state index in full system basis set
    :param j: state index in full system basis set
    include coupling between photon and detector.
    :return:
    '''
    coupling_value = -1.0  # this coupling value will be substituted later.

    bright_mode_state_quantum_number_di = state_quantum_number_di[0]
    bright_mode_state_quantum_number_dj = state_quantum_number_dj[0]

    if bright_mode_state_quantum_number_di - bright_mode_state_quantum_number_dj == 1:
        # all quantum numbers in other modes need to be the same
        state_quantum_number_difference = np.sum(np.abs(state_quantum_number_di[1:] - state_quantum_number_dj[1:]))

        if state_quantum_number_difference == 0:
            self.pd_coupling_irow.append(i)
            self.pd_coupling_icol.append(j)

            # include this coupling in matrix and irow, icol.
            self.append_matrix_element(coupling_value, i, j)
            # lower triangular part.
            self.append_matrix_element(coupling_value, j, i)

            self.pd_coupling_num = self.pd_coupling_num + 1

            self.offdiag_param_num = self.offdiag_param_num + 1

        else:
            pass

def _construct_offdiag_photon_detector_coupling(self):
    '''
    construct off diagonal coupling between photon and detector
    :param self: class pointer
    :return:
    '''
    for i in range(self._basis_set_state_num):
        for j in range(i+1, self._basis_set_state_num):
            ss = self.pstate[i] - self.pstate[j]

            di1 = self.dstate1[i]
            di2 = self.dstate2[i]

            dj1 = self.dstate1[j]
            dj2 = self.dstate2[j]

            # photon state index : 0 <-> [0,0], 1 <-> [1,0], 2 <-> [0,1],
            # no coupling between states with photon state [1,0] and [0,1]
            if self.pstate[i] + self.pstate[j] == 3:
                continue

            # coupling between photon and detector 1:
            # photon state [0,0] with index 0 and photon state [1,0] with index 1.
            if ss == -1 and di1 != dj1 and di2 == dj2:
                self._include_photon_detector_coupling(self.detector1.get_basis_set_state_quantum_number(i),
                                                       self.detector1.get_basis_set_state_quantum_number(j),
                                                       i, j)

            # coupling between photon and detector 2:
            # photon state [0,0] with index 0 and photon state [0,1] with index 2:
            if ss == -2 and di1 == dj1 and di2 != dj2:
                self._include_photon_detector_coupling(self.detector2.get_basis_set_state_quantum_number(i),
                                                       self.detector2.get_basis_set_state_quantum_number(j),
                                                       i, j)













