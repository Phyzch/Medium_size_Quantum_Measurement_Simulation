'''
define
'''

import numpy as np
import include.search_quantum_number_module
import include.hamiltonian_class

class Detector(include.hamiltonian_class.Hamiltonian):

    from _detector_class_operation import detector_add_basis_set_state

    from _output_detector_class_information import (output_detector_initial_state_index,
                                                    show_mode_frequency, show_dof, show_initial_state_qn,
                                                    get_basis_set_state_quantum_number,
                                                    output_detector_anharmonic_coupling_state_pairs)

    from _construct_detector_Hamiltonian_diagonal import (construct_detector_Hamiltonian_diagonal_part ,
                                                          _construct_detector_Hamiltonian_diagonal_part ,
                                                          _construct_offdiag_detector_state_coupling)

    from _construct_detector_Hamiltonian_offdiagonal import construct_offdiag_coupling_value


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
        init_state_pos, exist = include.search_quantum_number_module.binary_search_qn_list(self._basis_set_state_qn_list, self._initial_state_qn)
        if not exist:
            raise NameError("Wrong . Initial state not in state space")

        # record the index of initial state in basis set.
        self._initial_state_index = init_state_pos

        # initialize the wave function.
        self._wave_function = np.zeros(self._basis_set_state_num, dtype = np.complex)
        self._wave_function[init_state_pos] = 1



