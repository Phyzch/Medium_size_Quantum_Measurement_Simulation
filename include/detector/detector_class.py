import numpy as np
import include.search_qn_module
import include.hamiltonian_class

class Detector(include.hamiltonian_class.Hamiltonian):

    def __init__(self , *args):
        include.hamiltonian_class.Hamiltonian.__init__(self)

        dof, frequency, nmax, initial_state_qn, energy_window_for_basis_set, energy_window_for_coupling = args

        # dof: number of degree of freedoms for vibrational modes
        # frequency: vibrational mode frequencies.
        # nmax: maximum number of quanta included in each vibrational modes.
        # energy_window_for_basis_set: energy window for constructing basis set for detector Hamiltonian.
        self._dof = dof
        self._frequency = frequency
        self._nmax = nmax
        self._energy_window_for_basis_set_state = energy_window_for_basis_set

        # define the constraints we put in for constructing anharmonic coupling between vibrational states.
        # energy_window_for_coupling: only states with energy difference smaller than energy window is
        #                                            coupled to each other
        # qn_diff_cutoff: only states with 1-norm quantum number distance smaller than qn_diff_cutoff is
        #                                coupled to each other. 1-norm: \sum |n_i - n_j|
        self._energy_window_for_coupling = energy_window_for_coupling
        self._qn_diff_cutoff = 4

        # initial_state_qn : initial vibrational states of detector.
        # initial_state_index: index of the initial state in the basis set.
        self._initial_state_qn = np.copy(initial_state_qn)
        self._initial_state_index = -1

        # quantum number of basis set states.
        self._basis_set_state_qn_list = []

        # wave function array.
        self._wave_function = []


    # ------- initialize wave func ---------
    def initialize_wave_function(self):
        '''
        initialize wave function.
        The initial state is one state in the basis set.
        The quantum number (q.n.) of the initial state is defined by self.initial_state
        :return:
        '''
        init_state_pos, exist = include.search_qn_module.binary_search_qn_list(self._basis_set_state_qn_list, self._initial_state_qn)
        if not exist:
            raise NameError("Wrong . Initial state not in state space")

        # record the index of initial state in basis set.
        self._initial_state_index = init_state_pos

        # initialize the wave function.
        self._wave_function = np.zeros(self._basis_set_state_num, dtype = np.complex)
        self._wave_function[init_state_pos] = 1

    # ----- add basis set state to detector ----------------------------------------
    def detector_add_basis_set_state(self, state_quantum_number, state_energy, position):
        '''
        add basis set state to the list: _basis_set_state_qn_list, _basis_set_state_energy_list
        :param state_quantum_number: quantum number of basis set state to be added.
        :param state_energy: energy of the basis set state to be added.
        :param position: position to insert the quantum number of state and energy of the state.
        :return: None
        '''
        quantum_number_copy = np.copy(state_quantum_number)
        quantum_number_copy = quantum_number_copy.astype(int)

        # add qn to qn_list. add energy to energy_list.
        self._basis_set_state_qn_list.insert(position, quantum_number_copy)
        self._basis_set_state_energy_list.insert(position, state_energy)


    '''
    part I of constructing detector Hamiltonian. 
    We construct : 1. state 2. Hamiltonian diagonal part  3. irow, icol for coupling between states in Hamiltonian. 
    in part II, we will add strength of coupling between states (get from Genetic algorithm)
    '''
    def construct_detector_Hamiltonian_structure(self):
        '''
        decide the basis set states and how they are coupled to each other.
        off-diagonal coupling should read from Genetic algorithm part
        Here we only proceed to knowing off-digonal element numbers.
        We will need to output this number and read coupling outside class
        :return:
        '''

        # construct state's Hamiltonian diagonal part
        self._construct_detector_Hamiltonian_diagonal_part()

        # calculate state's anharmonic coupling
        self._construct_offdiag_detector_state_coupling()

    def _construct_detector_Hamiltonian_diagonal_part(self):
        '''
        construct basis set state and diagonal part of Hamiltonian for detectors.
        :return:
        '''
        assert len(self._initial_state_qn) == self._dof, print("Wrong. initial state doesn't have right dof")

        assert len(self._frequency) == self._dof, print("Wrong. frequency array does not have right dof")

        assert len(self._nmax) != self._dof, print("Wrong. nmax array does not right dof")

        quantum_number = np.zeros(self._dof)
        quantum_number[0] = -1

        initial_state_qn = np.array(self._initial_state_qn)
        initial_state_energy = np.sum(initial_state_qn * self._frequency)

        # Define a loop to go through states available in state space:
        exit_loop_bool = False
        while 1:
            for i in range(self._dof):
                quantum_number[i] = quantum_number[i] + 1
                if quantum_number[i] <= self._nmax[i]:
                    break
                if quantum_number[self._dof - 1] > self._nmax[self._dof - 1]:
                    exit_loop_bool = True

                quantum_number[i] = 0

            # exit the while(1) cycle
            if exit_loop_bool:
                break

            # --------- Check if this state is outside energy window, if so , jump to valid state -------------
            energy = np.sum(quantum_number * self._frequency)
            if energy > self._energy_window_for_basis_set_state + initial_state_energy:
                k = 0
                # jump to next state whose energy is smaller than initial_state_energy + energy_window
                while quantum_number[k] == 0:
                    quantum_number[k] = self._nmax[k]
                    k = k + 1
                    if k >= self._dof:
                        break

                if k < self._dof:
                    quantum_number[k] = self._nmax[k]

                continue
            # -----------------------------------

            # now put this state into state_mode_list which is ordered.
            position, exist = include.search_qn_module.binary_search_qn_list(self._basis_set_state_qn_list,
                                                                                         quantum_number)
            if not exist:
                self.detector_add_basis_set_state(quantum_number, energy, position)

        self._basis_set_state_num = len(self._basis_set_state_energy_list)

        # construct diagonal part of the hamiltonian.
        self._add_hamiltonian_diagonal_part()

    def _construct_offdiag_detector_state_coupling(self):
        '''
        As off-diagonal coupling strength is read from Genetic algorithm.
        we should calculate off-diagonal coupling number and output it
        :return:
        '''

        offdiag_hamiltonian_init_value = -1.0

        for i in range(self._basis_set_state_num):
            for j in range(i + 1, self._basis_set_state_num):
                state1_qn = self._basis_set_state_qn_list[i]
                state2_qn = self._basis_set_state_qn_list[j]

                # 1-norm distance between state i, j in state space
                deln = np.abs(np.array(state1_qn) - np.array(state2_qn))
                qn_num_diff = np.sum(deln)

                # ---- check diff mode_num_diff ---------
                assert qn_num_diff != 0, "Error. two different state in detector have same quantum number \n"

                if qn_num_diff == 2:
                    # higher order coupling is at least cubic term. thus q.n. diff = 2 corresponds to quartic term
                    qn_num_diff = 4
                    for k in range(self._dof):
                        if deln[k] == 1:
                            deln[k] = 2
                        if deln[k] == 2:
                            deln[k] = 4

                if qn_num_diff == 1:
                    # higher order coupling is at least cubic term. thus q.n. diff = 1 corresopnds to cubic term
                    qn_num_diff = 3
                    for k in range(self._dof):
                        if deln[k] == 1:
                            deln[k] = 3
                # ---------- check diff qn_num_diff -----------
                if qn_num_diff <= self._qn_diff_cutoff:
                    energy1 = self._basis_set_state_energy_list[i]
                    energy2 = self._basis_set_state_energy_list[j]
                    energy_diff = np.abs(energy1 - energy2)

                    # usually we will use cutoff criteria V / \Delta E. But now V should be sampled from Genetic algorithm
                    # Here instead we use energy window cutoff for criteria of anharmonic coupling.
                    # We require two state connected should be near in energy
                    if energy_diff <= self._energy_window_for_coupling:
                        # here we append off-diagonal hamiltonian element
                        # (independent off-diagonal element, if we include (i,j), we skip (j,i)) to the hamiltonian matrix.
                        # but the matrix element value is un-defined.
                        self._offdiagonal_coupling_num = self._offdiagonal_coupling_num + 1
                        self.append_matrix_element(offdiag_hamiltonian_init_value, i, j)

        self._mat_num = self._offdiagonal_coupling_num + self._basis_set_state_num

    '''
    construct off-diagonal coupling value of the Hamiltonian.
    '''

    def construct_offdiag_coupling_value(self, offdiag_coupling_element_list):
        '''
        Now we get value of off-diagonal matrix element.
        The row and column index of coupling matrix element is defined in _construct_diagonal_Hamiltonian.py
        :return:
        '''

        # read off diagonal coupling element. deep copy to avoid encapsulate the data.
        self._offdiag_coupling_element_list = offdiag_coupling_element_list.copy()

        if type(self._offdiag_coupling_element_list) == np.ndarray:
            self._offdiag_coupling_element_list = self._offdiag_coupling_element_list.tolist()

        # construct offdiagonal coupling element
        assert len(self._offdiag_coupling_element_list) == self._offdiagonal_coupling_num, \
            'off-diagonal coupling elements input from Genetic_algorithm do not have right length'

        # dirow, dicol is already add in list in calculate_offdiag_coupling_num(self). d_H here is list.
        assert(type(self._mat) == list and type(self._offdiag_coupling_element_list) == list)
        for i in range(self._offdiagonal_coupling_num):
            self._mat[i] = self._offdiag_coupling_element_list[i + self._basis_set_state_num]


    '''
    output information about detector Hamiltonian. 
    This only includes information specific to the detector Hamiltonian class.
    information specific to general sparse Hamiltonian matrix is defined in hamiltonian_class.py
    '''

    def output_detector_initial_state_index(self):
        '''
        output the index of initial state.
        :return:
        '''
        return self._initial_state_index


    def show_mode_frequency(self):
        '''
        return frequency of the detector.
        This function enforces the encapsulation of the frequency data.
        :return:
        '''
        return self._frequency.copy()


    def show_dof(self):
        '''

        :param self:
        :return:
        '''
        return self._dof


    def show_initial_state_qn(self):
        '''
        return initial state's quantum number of the detector.
        This function enforces the encapsulation of the initial state qn data.
        :return:
        '''
        return self._initial_state_qn.copy()


    def get_basis_set_state_quantum_number(self,i):
        '''
        return quantum number of basis set state for state index i.
        :param i: index for the element.
        :return:
        '''
        return self._basis_set_state_qn_list[i].copy()


    def output_detector_anharmonic_coupling_state_pairs(self):
        '''
        output information about detector state coupling in form of [state_qn1 , state_qn2]
        :return:
        '''
        # list for pair of vibrational states that couple to each other [state_qn1, state_qn2]
        state_qn_pair_for_anharmonic_coupling = []

        # going through the anharmonic coupling terms.
        for i in range(self._basis_set_state_num, self._mat_num):
            state_qn1 = self._basis_set_state_qn_list[self.get_irow(i)].tolist()
            state_qn2 = self._basis_set_state_qn_list[self.get_icol(i)].tolist()

            quantum_number_pair_of_anharmonic_coupling = [state_qn1, state_qn2]
            state_qn_pair_for_anharmonic_coupling.append(quantum_number_pair_of_anharmonic_coupling)

        # output the info of anharmonic coupling.
        print("detector Coupling: ")
        list_len = len(state_qn_pair_for_anharmonic_coupling)

        for i in range(list_len):
            print(state_qn_pair_for_anharmonic_coupling[i])