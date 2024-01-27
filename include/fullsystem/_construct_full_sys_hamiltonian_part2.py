import numpy as np
'''
part II of constructing Hamiltonian
We do the following : 1. revert Hamiltonian to diagonal form
                      2. read off-diagonal coupling strength from Genetic algorithm and add in Hamiltonian
'''

def construct_full_system_Hamiltonian_part2(self, offdiagonal_coupling_list):
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
    # the order we change hamiltonian value is the same as we construct hamiltonian matrix as in construct_full_sys_hamiltonian_part1.py
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
    d1_offdiag_coupling_num = len(self.d1_coupling_index)
    d2_offdiag_coupling_num = len(self.d2_coupling_index)

    # for detector 1:
    d1_hamiltonian_matrix_index = self._basis_set_state_num # used to update the hamiltonian of detector 1 on full system basis set.

    for i in range(d1_offdiag_coupling_num):
        d1_index = self.d1_coupling_index[i]
        anharmonic_coupling_value = self.detector1.get_matrix_value(d1_index)

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
    d2_hamiltonian_matrix_index = self._basis_set_state_num # used to update the hamiltonian of detector 2 on full system basis set.
    for i in range(d2_offdiag_coupling_num):
        d2_index = self.d2_coupling_index[i]
        anharmonic_coupling_value = self.detector2.get_matrix_value(d2_index)
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





