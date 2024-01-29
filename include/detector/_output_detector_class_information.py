import numpy as np
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
        state_qn1 = self._basis_set_state_qn_list[self._d_Hamiltonian.get_irow(i)].tolist()
        state_qn2 = self._basis_set_state_qn_list[self._d_Hamiltonian.get_icol(i)].tolist()

        quantum_number_pair_of_anharmonic_coupling = [state_qn1, state_qn2]
        state_qn_pair_for_anharmonic_coupling.append(quantum_number_pair_of_anharmonic_coupling)

    # output the info of anharmonic coupling.
    print("detector Coupling: ")
    list_len = len(state_qn_pair_for_anharmonic_coupling)

    for i in range(list_len):
        print(state_qn_pair_for_anharmonic_coupling[i])