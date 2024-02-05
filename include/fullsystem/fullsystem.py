'''
This class serve as creating quantum system which consist of
two medium-size-subsystem (we call it detector) and one small-size subsystem (we call it photon)
The coupling parameter between states is optimized using Genetic algorithm if we want to search best parameter to give best localization.

To use this class:
1. construct fullsystem diagonal hamiltonian_part
2.  output_offdiagonal_parameter_number(self) to tell Genetic algorithm the number of off-diagonal parameter we need to feed
3. Then run construct_full_system_Hamiltonian_offdiagonal_part(self , offdiagonal_coupling_list)
to update off-diagonal coupling of the full system
'''

import numpy as np
import include.detector.detector_class
import include.hamiltonian_class
from ._evolve_wave_func import wave_func_sum, check_energy_conservation


class FullSystem(include.hamiltonian_class.Hamiltonian):

    def __init__(self, detector_1_parameter, detector_2_parameter, energy_window_for_basis_set_state, photon_energy,
                 initial_photon_wave_function, time_duration=5000, output_time_step=10):
        super().__init__()

        self.offdiag_param_list = None
        self.time_duration = time_duration
        self.output_time_step = output_time_step

        # Instantiate the detector class for two detectors.
        # Energy_window for each detector is contained in detector_parameter.
        self.detector1 = include.detector.detector_class.Detector(*detector_1_parameter)
        self.detector2 = include.detector.detector_class.Detector(*detector_2_parameter)

        # Initialize wave function for photon
        self.initial_photon_wave_function = initial_photon_wave_function
        # Initial energy of photon + detector 1 + detector 2. we will compute it later in the class function.
        self.initial_energy = 0

        # For every state in full system, it composed of system state, detector 1 's state, detector 2's state.
        # state_mode_list contain information about : [ photon_mode, detector1_mode, detector2_mode ]
        self.photon_state_index = []
        self.detector1_state_index = []
        self.detector2_state_index = []
        self._basis_set_state_qn_list = []

        # System state is [[00], [01], [10]], thus system's state number is 3.
        self.photon_state_num = 3
        # Energy of photon
        self.init_photon_energy = photon_energy
        self.photon_state_energy = [0, photon_energy, photon_energy]
        self.photon_state_qn = [[0, 0], [1, 0], [0, 1]]

        # Energy window is used to construct state in full system relative to initial state.
        self.energy_window_for_basis_set_state = energy_window_for_basis_set_state

        self._offdiagonal_coupling_num = 0  # we only count independent off-diagonal coupling # here.

        # Off-diagonal_parameter_number is output to Genetic algorithm for optimization of the energy localization.
        # Off_diagonal_parameter between photon and detector
        self.pd_offdiag_param = []
        # coupling index between detector and photon
        self.pd_coupling_irow = []
        self.pd_coupling_icol = []
        self.pd_coupling_num = 0

        # Off_diagonal_parameter between detector and detector.
        self.dd_offdiag_param = []
        # coupling index between detector and detector.
        self.dd_coupling_irow = []
        self.dd_coupling_icol = []
        self.dd_coupling_num = 0

        # anharmonic coupling index in detector 1 hamiltonian and detector 2 hamiltonian.
        # use this to refer back to detector hamiltonian.
        self.d1_intra_detector_coupling_index_in_detector_hamiltonian = []
        self.d2_intra_detector_coupling_index_in_detector_hamiltonian = []

        # wave function for photon:
        self.photon_wave_func = np.zeros(self.photon_state_num)
        self.photon_wave_func[1] = 1 / np.sqrt(2)
        self.photon_wave_func[2] = 1 / np.sqrt(2)
        # wave function for full system:
        self._wave_function = []

        # hamiltonian for the photon and the detector on the full system basis set.
        self.photon_hamiltonian_in_full_basis_set = include.hamiltonian_class.Hamiltonian()

        self.detector1_hamiltonian_in_full_basis_set = include.hamiltonian_class.Hamiltonian()
        self.detector2_hamiltonian_in_full_basis_set = include.hamiltonian_class.Hamiltonian()

    def _full_system_add_basis_set_state(self, state_quantum_number, state_energy, position):
        '''
        add basis set state to the list : _basis_set_state_qn_list, _basis_set_state_energy_list
        :param state_quantum_number: quantum number of detector and photon of basis set state to be added.
        :param state_energy: energy of the basis set state
        :param position: position to insert the quantum number of state and energy of the state.
        :return:
        '''
        quantum_number_copy = np.copy(state_quantum_number)
        quantum_number_copy = quantum_number_copy.astype(int)

        # add qn to qn_list. add energy to energy_list.
        self._basis_set_state_qn_list.insert(position, quantum_number_copy)
        self._basis_set_state_energy_list.insert(position, state_energy)

    '''
    first part of constructing Hamiltonian. This only have to be called once.
    We do following : 1. compute energy of full_system  : self.__compute_initial_energy()
                      2. let detector construct Hamiltonian and coupling info: self.detector1/2.construct_detector_Hamiltonian_part1()
                      3. construct diagonal part of Hamiltonian and state for full_system : self.__construct_full_system_diagonal_Hamiltonian()
                      4. construct irow, icol for off-diagonal coupling between state : self.__construct_offdiag_dd_pd_coup() , self.__construct_intra_detector_coupling()
    '''

    def construct_full_system_hamiltonian_structure(self):
        '''
        construct hamiltonian for two detectors.
        Then construct basis set and diagonal part of full system's hamiltonian.
        then construct row and column index for off-diagonal coupling for full system's hamiltonian.
        :return:
        '''
        # construct basis set for detector state and compute # of off-diagonal couplings.
        self.detector1.construct_detector_hamiltonian_structure()
        self.detector2.construct_detector_hamiltonian_structure()

        # construct basis set and diagonal part of the hamiltonian.
        self.construct_full_system_hamiltonian_diagonal_part()
        # construct row and column index for off-diagonal coupling on full system basis set.
        self.construct_full_system_off_diagonal_coupling_index()

    def construct_full_system_hamiltonian_diagonal_part(self):
        '''
        compute initial energy of the full system.
        construct basis set for full system.
        :return:
        '''
        self._compute_initial_state_energy()

        # construct basis set for full system.
        # also construct diagonal part of Hamiltonian
        self._construct_full_system_basis_set()

        # shift Hamiltonian's diagonal part by initial energy to speed up propagation of wave function.
        self._shift_Hamiltonian()



    def construct_full_system_off_diagonal_coupling_index(self):
        '''
        compute row and column index for off-diagonal coupling.
        This includes 3 parts: (1) detector-detector coupling (2) photon-detector coupling
        (3) intra-detector anharmonic coupling
        :return:
        '''
        # the independent off-diagonal number we need to optimize in Genetic algorithm.
        self.offdiag_param_num = self.offdiag_param_num + self.detector1.show_offdiag_matrix_num()
        self.offdiag_param_num = self.offdiag_param_num + self.detector2.show_offdiag_matrix_num()

        # compute the parameter number and row, col indexes for coupling between two detectors
        self._construct_offdiag_detector_detector_coupling()

        # compute the parameter number and row, col indexes for coupling between photon and detector
        self._construct_offdiag_photon_detector_coupling()

        # compute position of intra-detector coupling
        self._construct_intra_detector_coupling()

    def _compute_initial_state_energy(self):
        '''
        compute full system initial state's energy.
        :return:
        '''
        self.initial_energy = 0

        d1_energy = np.sum(np.array(self.detector1.show_mode_frequency()) * np.array(
            self.detector1.show_initial_state_qn()))

        d2_energy = np.sum(np.array(self.detector2.show_mode_frequency()) * np.array(
            self.detector2.show_initial_state_qn()))

        self.initial_energy = self.init_photon_energy + d1_energy + d2_energy

    def _construct_full_system_basis_set(self):
        '''
        construct state and diagonal part of Hamiltonian.
        impose energy window : states included should satisfy : |E - E_init | <= energy_window
        :return:
        '''
        basis_set_state_index = 0

        photon_state_num = self.photon_state_num
        detector1_state_num = self.detector1.show_state_num()
        detector2_state_num = self.detector2.show_state_num()

        for i in range(photon_state_num):
            for j in range(detector1_state_num):
                for k in range(detector2_state_num):
                    energy = self.photon_state_energy[i] + self.detector1.get_basis_set_state_energy(j) + \
                             self.detector2.get_basis_set_state_energy(k)

                    if abs(energy - self.initial_energy) <= self.energy_window_for_basis_set_state:
                        # record photon state and detector state index for state in full system.
                        self.photon_state_index.append(i)
                        self.detector1_state_index.append(j)
                        self.detector2_state_index.append(k)

                        state_quantum_number = [self.photon_state_qn[i],
                                                self.detector1.get_basis_set_state_quantum_number(j).tolist(),
                                                self.detector2.get_basis_set_state_quantum_number(k).tolist()]

                        # add basis set state qn and energy to list.
                        self._full_system_add_basis_set_state(state_quantum_number, energy, basis_set_state_index)

                        # Hamiltonian for photon, d1, d2
                        self.photon_hamiltonian_in_full_basis_set.append_matrix_element(self.photon_state_energy[i],
                                                                                        basis_set_state_index,
                                                                                        basis_set_state_index)

                        self.detector1_hamiltonian_in_full_basis_set.append_matrix_element(
                            self.detector1.get_basis_set_state_energy(j),
                            basis_set_state_index, basis_set_state_index)

                        self.detector2_hamiltonian_in_full_basis_set.append_matrix_element(
                            self.detector2.get_basis_set_state_energy(k),
                            basis_set_state_index, basis_set_state_index)

                        basis_set_state_index = basis_set_state_index + 1

        self._basis_set_state_num = basis_set_state_index

        # update diagonal Hamiltonian and Hamiltonian itself.
        self._add_basis_set_energy_to_hamiltonian_diagonal_part()

    def _shift_Hamiltonian(self):
        '''
        shift Hamiltonian by initial state energy : E.
        Initial energy E will only contribute to a phase when we evovle wave function.
        '''
        for i in range(self._basis_set_state_num):
            self._mat[i] = self._mat[i] - self.initial_energy

    def _construct_intra_detector_coupling_submodule(self, i, j, di, dj, detector,
                                                     detector_hamiltonian_in_full_basis_set,
                                                     intra_detector_coupling_index_in_detector_hamiltonian):
        '''
        :param i, j: state index in full_system
        :param di, dj : state index in detector .
        :return:
        '''
        coupling_value = - 1.0  # this coupling value will be substituted later.
        detector_state_num = detector.show_state_num()
        detector_mat_num = detector.show_mat_num()

        for k in range(detector_state_num, detector_mat_num):
            # k is off-diagonal matrix index in detector's Hamiltonian
            if detector.get_irow(k) == di and detector.get_icol(k) == dj:
                # add intra-detector anharmonic coupling to detector hamiltonian in full basis set.
                detector_hamiltonian_in_full_basis_set.append_matrix_element(coupling_value, i, j)
                detector_hamiltonian_in_full_basis_set.append_matrix_element(coupling_value, j, i)

                # index for anharmonic coupling in detector hamiltonian is recorded for reference.
                # when we assign value of anharmonic coupling to hamiltonian, we will need this info.
                intra_detector_coupling_index_in_detector_hamiltonian.append(k)

                # for evolve wave function, we need to record off-diagonal matrix element symmetrically.
                self.append_matrix_element(coupling_value, i, j)
                self.append_matrix_element(coupling_value, j, i)

                break

    def _construct_intra_detector_coupling(self):
        # first include anharmonic coupling in detector 1.
        for i in range(self._basis_set_state_num):
            for j in range(i + 1, self._basis_set_state_num):
                ss = self.photon_state_index[i] - self.photon_state_index[j]

                # i , j stands for different state.  1, 2 stand for detector 1 and detector 2
                di1 = self.detector1_state_index[i]
                di2 = self.detector2_state_index[i]
                dj1 = self.detector1_state_index[j]
                dj2 = self.detector2_state_index[j]

                # coupling in detector 1
                if ss == 0 and di1 != dj1 and di2 == dj2:
                    self._construct_intra_detector_coupling_submodule(i, j, di1, di2, self.detector1,
                                                                      self.detector1_hamiltonian_in_full_basis_set,
                                                                      self.d1_intra_detector_coupling_index_in_detector_hamiltonian)

        # then include anharmonic coupling in detector 2.
        for i in range(self._basis_set_state_num):
            for j in range(i + 1, self._basis_set_state_num):
                ss = self.photon_state_index[i] - self.photon_state_index[j]

                # i , j stands for different state.  1, 2 stand for detector 1 and detector 2
                di1 = self.detector1_state_index[i]
                di2 = self.detector2_state_index[i]
                dj1 = self.detector1_state_index[j]
                dj2 = self.detector2_state_index[j]

                # coupling in detector2
                if ss == 0 and di1 == dj1 and di2 != dj2:
                    self._construct_intra_detector_coupling_submodule(i, j, di2, dj2, self.detector2,
                                                                      self.detector2_hamiltonian_in_full_basis_set,
                                                                      self.d2_intra_detector_coupling_index_in_detector_hamiltonian)

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
        coupling_value = - 1.0  # this coupling value will be substituted later.
        # examine if sum|n_i - n_j| = 1, as the detector detector coupling form we choose is to exchange 1 quantum number
        # (a_i a_j^{+} + a_i^{+} a_j)
        d1_succeed = examine_detector_detector_coupling(self.detector1.get_basis_set_state_quantum_number(di1),
                                                        self.detector1.get_basis_set_state_quantum_number(dj1))
        d2_succeed = examine_detector_detector_coupling(self.detector2.get_basis_set_state_quantum_number(di2),
                                                        self.detector2.get_basis_set_state_quantum_number(dj2))

        if d1_succeed and d2_succeed:
            self.dd_coupling_irow.append(i)
            self.dd_coupling_icol.append(j)
            # add off-diagonal matrix element to Hamiltonian.
            self.append_matrix_element(coupling_value, i, j)
            # lower triangular part of Hamiltonian.
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
            for j in range(i + 1, self._basis_set_state_num):
                ss = self.photon_state_index[i] - self.photon_state_index[j]
                # index for detector 1 and detector 2 for state i.
                di1 = self.detector1_state_index[i]
                di2 = self.detector2_state_index[i]
                # index for detector 1 and detector 2 for state j
                dj1 = self.detector1_state_index[j]
                dj2 = self.detector2_state_index[j]

                # photon state index : 0 <-> [0,0], 1 <-> [1,0], 2 <-> [0,1],
                # no coupling between states with photon state [1,0] and [0,1]
                if self.photon_state_index[i] + self.photon_state_index[j] == 3:
                    continue

                # coupling between detector 1 and detector 2
                if ss == 0 and di1 != dj1 and di2 != dj2:
                    self._include_detector_detector_coupling(di1, dj1, di2, dj2, i, j)

    def _include_photon_detector_coupling(self, state_quantum_number_di, state_quantum_number_dj, i, j):
        '''
        :param: state_quantum_number_di: basis set state index for detector state in detector (1 or 2) for full system state i.
        :param state_quantum_number_dj: basis set state index for detector state in detector (1 or 2) for full system state j
        :param i: state index in full system basis set
        :param j: state index in full system basis set
        include coupling between photon and detector.
        :return:
        '''
        coupling_value = -1.0  # this coupling value will be substituted later.

        # bright mode is the vibrational mode that is first excited when detector absorb energy from photon.
        bright_mode_state_quantum_number_di = state_quantum_number_di[0]
        bright_mode_state_quantum_number_dj = state_quantum_number_dj[0]

        if bright_mode_state_quantum_number_di - bright_mode_state_quantum_number_dj == 1:
            # all quantum numbers in other modes (except bright mode) need to be the same
            state_quantum_number_difference = np.sum(np.abs(state_quantum_number_di[1:] - state_quantum_number_dj[1:]))

            if state_quantum_number_difference == 0:
                self.pd_coupling_irow.append(i)
                self.pd_coupling_icol.append(j)

                # include this coupling in matrix and row, col index list.
                self.append_matrix_element(coupling_value, i, j)
                # lower triangular part.
                self.append_matrix_element(coupling_value, j, i)
                # number of photon-detector couplings.
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
            for j in range(i + 1, self._basis_set_state_num):
                ss = self.photon_state_index[i] - self.photon_state_index[j]
                # state index in detector 1 and detector 2 for full system state i
                di1 = self.detector1_state_index[i]
                di2 = self.detector2_state_index[i]
                # state index in detector 1 and detector 2 for full system state j.
                dj1 = self.detector1_state_index[j]
                dj2 = self.detector2_state_index[j]

                # photon state index : 0 <-> [0,0], 1 <-> [1,0], 2 <-> [0,1],
                # no coupling between states with photon state [1,0] and [0,1]
                if self.photon_state_index[i] + self.photon_state_index[j] == 3:
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

    '''
    part II of constructing Hamiltonian
    We do the following : 1. revert Hamiltonian to diagonal form
                          2. read off-diagonal coupling strength from Genetic algorithm and add in Hamiltonian
    '''

    def construct_full_system_Hamiltonian_offdiagonal_part(self, offdiagonal_coupling_list):
        '''
        After we read offdiagonal parameter from Genetic algorithm, we do this part.
        offdiagonal_coupling_list : size [self.offdiagonal coupling num]
        :return:
        '''
        # Then read offdiagonal coupling parameter
        self.read_offdiag_coupling_element(offdiagonal_coupling_list)

        # full system construct Hamiltonian using detector's Hamiltonian & coupling between p-d and d-d
        self._construct_full_system_offdiag_coupling()

    def _construct_full_system_offdiag_coupling(self):
        '''

        :param self:
        :return:
        '''
        # assign off-diagonal coupling value for detector-detector coupling matrix element.
        assert self.dd_coupling_num == len(self.dd_offdiag_param), "inter-detector coupling number is wrong."

        hamiltonian_matrix_index = self._basis_set_state_num
        # the order we change hamiltonian value is the same as we construct hamiltonian matrix as in construct_full_system_off_diagonal_coupling_index()
        # as we include both matrix element [i,j] and [j,i], we have to replace off-diagonal coupling in matrix twice.
        for i in range(self.dd_coupling_num):
            self._mat[hamiltonian_matrix_index] = self.dd_offdiag_param[i]
            hamiltonian_matrix_index = hamiltonian_matrix_index + 1
            self._mat[hamiltonian_matrix_index] = self.dd_offdiag_param[i]
            hamiltonian_matrix_index = hamiltonian_matrix_index + 1

        # assign off-diagonal coupling value for photon-detector coupling matrix element.
        assert self.pd_coupling_num == len(self.pd_offdiag_param), "photon-detector coupling number is wrong."

        for i in range(self.pd_coupling_num):
            self._mat[hamiltonian_matrix_index] = self.pd_offdiag_param[i]
            hamiltonian_matrix_index = hamiltonian_matrix_index + 1
            self._mat[hamiltonian_matrix_index] = self.pd_offdiag_param[i]
            hamiltonian_matrix_index = hamiltonian_matrix_index + 1

        # anharmonic coupling within detectors.
        d1_offdiag_coupling_num = len(self.d1_intra_detector_coupling_index_in_detector_hamiltonian)
        d2_offdiag_coupling_num = len(self.d2_intra_detector_coupling_index_in_detector_hamiltonian)

        # for detector 1:
        d1_hamiltonian_matrix_index = self._basis_set_state_num  # used to update the hamiltonian of detector 1 on full system basis set.

        for i in range(d1_offdiag_coupling_num):
            d1_index = self.d1_intra_detector_coupling_index_in_detector_hamiltonian[i]
            anharmonic_coupling_value = self.detector1.get_matrix_value(d1_index)   # the coupling value should be the same as in detector 1's hamiltonian.

            # update full system's hamiltonian.
            self._mat[hamiltonian_matrix_index] = anharmonic_coupling_value
            hamiltonian_matrix_index = hamiltonian_matrix_index + 1

            self._mat[hamiltonian_matrix_index] = anharmonic_coupling_value
            hamiltonian_matrix_index = hamiltonian_matrix_index + 1

            # update hamiltonian of detector 1 on full system basis set.
            self.detector1_hamiltonian_in_full_basis_set.replace_mat_value(d1_hamiltonian_matrix_index,
                                                                           anharmonic_coupling_value)
            d1_hamiltonian_matrix_index = d1_hamiltonian_matrix_index + 1

            self.detector1_hamiltonian_in_full_basis_set.replace_mat_value(d1_hamiltonian_matrix_index,
                                                                           anharmonic_coupling_value)
            d1_hamiltonian_matrix_index = d1_hamiltonian_matrix_index + 1

        # for detector 2:
        d2_hamiltonian_matrix_index = self._basis_set_state_num  # used to update the hamiltonian of detector 2 on full system basis set.
        for i in range(d2_offdiag_coupling_num):
            d2_index = self.d2_intra_detector_coupling_index_in_detector_hamiltonian[i]
            anharmonic_coupling_value = self.detector2.get_matrix_value(d2_index)    # the coupling value should be the same as in detector 2's hamiltonian.

            # update full system's hamiltonian.
            self._mat[hamiltonian_matrix_index] = anharmonic_coupling_value
            hamiltonian_matrix_index = hamiltonian_matrix_index + 1

            self._mat[hamiltonian_matrix_index] = anharmonic_coupling_value
            hamiltonian_matrix_index = hamiltonian_matrix_index + 1

            # update hamiltonian of detector 2 on full system basis set.
            self.detector2_hamiltonian_in_full_basis_set.replace_mat_value(d2_hamiltonian_matrix_index,
                                                                           anharmonic_coupling_value)
            d2_hamiltonian_matrix_index = d2_hamiltonian_matrix_index + 1

            self.detector2_hamiltonian_in_full_basis_set.replace_mat_value(d2_hamiltonian_matrix_index,
                                                                           anharmonic_coupling_value)
            d2_hamiltonian_matrix_index = d2_hamiltonian_matrix_index + 1

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

                wave_function_list.append(self._wave_function)

                photon_energy = self._evaluate_photon_energy()

                d1_energy = self._evaluate_detector_energy(self.detector1_hamiltonian_in_full_basis_set)

                d2_energy = self._evaluate_detector_energy(self.detector2_hamiltonian_in_full_basis_set)

                photon_energy_list.append(photon_energy)
                detector1_energy_list.append(d1_energy)
                detector2_energy_list.append(d2_energy)

                if abs(photon_energy - self.init_photon_energy) > 0.1:
                    raise NameError("Error for photon energy convergence")

                time_list.append(t)

            # evolve wave function. simple SUR algorithm: https://doi.org/10.1016/0009-2614(94)01474-A
            # real_part = real_part + H * dt * imag_part
            real_part_change = self.get_mat_array() * imag_part_wave_func[self.get_icol_array()] * time_step
            # use numba to speed up H = H + H_change. For numba, see : https://numba.pydata.org/
            real_part_wave_func = wave_func_sum(real_part_wave_func, real_part_change, self.get_irow_array())

            # imag_part = imag_part - H * dt * real_part
            imag_part_change = - self.get_mat_array() * real_part_wave_func[self.get_icol_array()] * time_step
            # use numba to speed up H = H + H_change.
            imag_part_wave_func = wave_func_sum(imag_part_wave_func, imag_part_change, self.get_irow_array())

            t = t + time_step

        detector1_energy_list = np.array(detector1_energy_list)
        detector2_energy_list = np.array(detector2_energy_list)
        photon_energy_list = np.array(photon_energy_list)
        time_list = np.array(time_list)

        # check energy conservation
        check_energy_conservation(time_step, time_list, detector1_energy_list, detector2_energy_list,
                                  photon_energy_list)

        return photon_energy_list, detector1_energy_list, detector2_energy_list, time_list

    def initialize_wave_function(self):
        '''
        initialize full system's initial wave function.
        :param self: class pointer
        :return:
        '''
        # initialize wave_func for d1, d2
        # assume initially detector1 and detector2 are in pure state.
        self.detector1.initialize_wave_function()
        self.detector2.initialize_wave_function()

        detector1_initial_state_index = self.detector1.output_detector_initial_state_index()
        detector2_initial_state_index = self.detector2.output_detector_initial_state_index()

        self._wave_function = np.zeros(self._basis_set_state_num, dtype=np.complex)

        for i in range(self._basis_set_state_num):
            if self.detector1_state_index[i] == detector1_initial_state_index and self.detector2_state_index[i] == detector2_initial_state_index:
                if self.photon_state_index[i] == 1:
                    self._wave_function[i] = self.initial_photon_wave_function[0]
                if self.photon_state_index[i] == 2:
                    self._wave_function[i] = self.initial_photon_wave_function[1]

    # ---------- evaluate photon , detector1, detector2 energy --------------
    def _evaluate_photon_energy(self):
        '''
        evaluate the energy of photon
        :return:
        '''
        # compute H * phi, here H is photon hamiltonian on full system's basis set.
        # H[irow, icol], here h_phi_1[i] is the value of (H * phi) [irow[i]]
        h_phi_1 = self.photon_hamiltonian_in_full_basis_set.get_mat_array() * self._wave_function[self.get_icol_array()]

        # compute H * phi
        h_phi_wave_function = np.zeros(self._basis_set_state_num, dtype=np.complex)
        h_phi_wave_function = wave_func_sum(h_phi_wave_function, h_phi_1,
                                            self.photon_hamiltonian_in_full_basis_set.get_irow_array())
        # compute <phi | H | phi>. photon energy.
        photon_energy = np.sum(np.real(np.conjugate(self._wave_function) * h_phi_wave_function))

        return photon_energy

    def _evaluate_detector_energy(self, detector_hamiltonian):
        '''
        compute energy of detector.
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

    def output_state_qn_number_list(self):
        '''
        output quantum num for state
        :return:
        '''
        print(self._basis_set_state_qn_list)

    def output_off_diagonal_coupling_state_pairs_info(self):
        '''

        :return:
        '''
        coupling_state_qn_list = []
        for i in range(self._basis_set_state_num, self._mat_num, 2):
            irow_index = self._irow[i]
            icol_index = self._icol[i]

            coupling_state_qn = [self._basis_set_state_qn_list[irow_index], self._basis_set_state_qn_list[icol_index]]

            coupling_state_qn_list.append(coupling_state_qn)

        list_len = len(coupling_state_qn_list)
        print("Coupling number:  " + str(list_len) + "\n")
        print("Coupling for state in full system: ")
        for i in range(list_len):
            print(coupling_state_qn_list[i])

    def read_offdiag_coupling_element(self, offdiagonal_coupling_list):
        '''
        read offdiagonal coupling element into fullsystem instance.
        :param offdiagonal_coupling_list: python list. off-diag coupling param.
        :return:
        '''
        self.offdiag_param_list = offdiagonal_coupling_list.copy()

        # intra-detector coupling matrix element in detector 1.
        begin_index = 0
        end_index = self.detector1.show_offdiag_matrix_num()
        # always deep-copy the array to ensure encapsulation of the data.
        d1_offdiag_param = offdiagonal_coupling_list[begin_index: end_index].copy()
        # construct anharmonic (off-diagonal) coupling in detector 1.
        self.detector1.construct_offdiag_coupling_value(d1_offdiag_param)

        # intra-detector coupling matrix element in detector 2.
        begin_index = end_index
        end_index = end_index + self.detector2.show_offdiag_matrix_num()
        # always deep-copy the array to ensure encapsulation of the data.
        d2_offdiag_param = offdiagonal_coupling_list[begin_index: end_index].copy()
        # construct anharmonic (off-diagonal) coupling in detector 2
        self.detector2.construct_offdiag_coupling_value(d2_offdiag_param)

        # detector-detector coupling matrix element
        begin_index = end_index
        end_index = begin_index + self.dd_coupling_num
        self.dd_offdiag_param = offdiagonal_coupling_list[begin_index: end_index].copy()

        # photon-detector coupling matrix element.
        begin_index = end_index
        end_index = begin_index + self.pd_coupling_num
        self.pd_offdiag_param = offdiagonal_coupling_list[begin_index: end_index].copy()



def examine_detector_detector_coupling(state_quantum_number_di, state_quantum_number_dj):
    '''
    check whether the quantum number of the detector for the state di and the state dj only differ by one.
    :param: state_quantum_number_di : quantum number of detector states dj
    :param: state_quantum_number_dj : quantum number of detectors states di
    :return:
    '''
    quantum_number_difference_sum = np.sum(np.abs(state_quantum_number_di - state_quantum_number_dj))
    if quantum_number_difference_sum == 1:
        succeed = True
    else:
        succeed = False
    return succeed