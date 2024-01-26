'''
operations to the detector class
'''
import numpy as np

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


def _detector_add_hamiltonian_diagonal_part(self):
    '''
    construct diagonal part of the hamiltonian.
    :return:
    '''
    # construct detector Hamiltonian diagonal part.
    for i in range(self._basis_set_state_num):
        self.append_matrix_element(self._basis_set_state_energy_list[i], i, i)

    # record diagonal part of Hamiltonian
    self._diagonal_mat = self._mat.copy()