'''
define
'''

import numpy as np
from include.Constructing_state_module import binary_search_qn_list
from include.Hamiltonian_class import  Hamiltonian

class Detector(Hamiltonian):

    # fixme: what is the relation between hamiltonian class and detector class? It should be inheritence.

    from _detector_class_member_operation import detector_add_basis_set_state, _detector_add_hamiltonian_diagonal_part

    from _output_detector_class_information import (output_detector_initial_state_index,
                                                    output_detector_anharmonic_coupling_state_pairs)

    from _construct_detector_Hamiltonian_part1 import (construct_detector_Hamiltonian_part1 ,
                                                       _construct_detector_Hamiltonian_diagonal_part ,
                                                       _construct_offdiag_detector_state_coupling)

    from _construct_detector_Hamiltonian_part2 import construct_offdiag_coupling_value, reverse_dmat_diag_form


    def __init__(self , *args):
        Hamiltonian.__init__(self)

        dof, frequency, nmax, initial_state_qn, energy_window_for_basis_set, energy_window_for_coupling = args

        self.define_vibrational_modes_and_basis_set_condition(dof, frequency, nmax, energy_window_for_basis_set)

        self.define_anharmonic_coupling_constraints(energy_window_for_coupling)

        self.define_initial_vib_states_of_detector(initial_state_qn)

        self._basis_set_state_qn_list = []

        # wave function array.
        self._wave_function = []

    def define_vibrational_modes_and_basis_set_condition(self, dof, frequency, nmax, energy_window_for_basis_set):
        '''
        code to initialize vibrational modes that constitute detectors.
        :param dof:  # of degree of freedoms for vibrational modes
        :param frequency: vibrational mode frequencies.
        :param nmax: maximum number of quanta included in each vibrational modes.
        :param energy_window_for_basis_set: energy window for constructing basis set for detector Hamiltonian.
        :return:
        '''
        self._dof = dof
        self._frequency = frequency
        self._nmax = nmax
        self._energy_window = energy_window_for_basis_set

    def define_anharmonic_coupling_constraints(self, energy_window_for_coupling):
        '''
        define the constraints we put in for constructing anharmonic coupling between vibrational states.
        :param energy_window_for_coupling: only states with energy difference smaller than energy window is
                                           coupled to each other
        :param qn_diff_cutoff: only states with 1-norm quantum number distance smaller than qn_diff_cutoff is
                               coupled to each other. 1-norm: \sum |n_i - n_j|
        :return:
        '''
        # we also have energy window cutoff for states to couple with each other
        self._energy_window_for_coupling = energy_window_for_coupling

        self._qn_diff_cutoff = 4

    def define_initial_vib_states_of_detector(self, initial_state_qn):
        '''

        :param initial_state_qn: initial vibrational states of detector.
        :return:
        '''
        # it's important to copy the array here to encapsulate the data.
        self._initial_state_qn = np.copy(initial_state_qn)
        self._initial_state_index = -1




    # ------- initialize wave func ---------
    def initialize_wave_function(self):
        '''
        initialize wave function.
        The initial state is one state in the basis set.
        The quantum number (q.n.) of the initial state is defined by self.initial_state
        :return:
        '''
        init_state_pos, exist = binary_search_qn_list(self._basis_set_state_qn_list, self._initial_state_qn)
        if not exist:
            raise NameError("Wrong . Initial state not in state space")

        # record the index of initial state in basis set.
        self._initial_state_index = init_state_pos

        # initialize the wave function.
        self._wave_function = np.zeros(self._basis_set_state_num, dtype = np.complex)
        self._wave_function[init_state_pos] = 1



