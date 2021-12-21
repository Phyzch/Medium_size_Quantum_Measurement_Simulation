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
    # First reverse matrix to contain only diagonal part.
    self._reverse_mat_diag_form()

    # Then read offdiagonal coupling parameter
    d1_offdiag_param, d2_offdiag_param = self.read_offdiag_coupling_element(offdiagonal_coupling_list)

    # each detector construct their hamiltonian
    self.detector1.construct_offdiag_mat(d1_offdiag_param)
    self.detector2.construct_offdiag_mat(d2_offdiag_param)

    # full system construct Hamiltonian using detector's Hamiltonian & coupling between p-d and d-d
    self._construct_full_system_offdiag_coupling()



def _reverse_mat_diag_form(self):
    # For each generation, we only have to update off-diagonal part .
    self.detector1.reverse_dmat_diag_form()
    self.detector2.reverse_dmat_diag_form()

    self.full_H.mat = self.full_H.diag_mat.copy()

    self.d1_H.mat = self.d1_H.diag_mat.copy()
    self.d2_H.mat = self.d2_H.diag_mat.copy()


def _construct_full_system_offdiag_coupling(self):
    # ---------- inline function ---------
    def construct_intra_d_coup(intra_d_coup_num, dmat_index, dmat, d_H):
        '''

        :param intra_d_coup_num: number of intra-detector coupling matrix element
        :param dmat_index: index of element in detector's Hamiltonian
        :param dmat: detector matrix
        :param d_H: Hamiltonian solely for detector in full matrix
        :return:
        '''
        for i in range(intra_d_coup_num):
            k = dmat_index[i]
            self.full_H.mat.append(dmat[k])
            # we also record lower trangular part
            self.full_H.mat.append(dmat[k])

            # construct Hamiltonian for d_H
            d_H.mat.append(dmat[k])
            # we also record lower trangular part
            d_H.mat.append(dmat[k])

    # ---------- inline function ----------

    pd_dd_coupling_num = len(self.pd_dd_coupling_irow)
    if (pd_dd_coupling_num != len(self.pd_dd_offdiag_param)):
        raise NameError("inter detector coupling number does not equal to parameter number read from Genetic algorithm")

    # order of adding element below is the same we construct irow, icol.
    # coupling between detector and photon (pd) . and detector between detector (dd)
    for i in range(pd_dd_coupling_num):
        self.full_H.mat.append(self.pd_dd_offdiag_param[i])
        self.full_H.mat.append(self.pd_dd_offdiag_param[i])

    # coupling in detector 1
    intra_d1_coupling_num = len(self.d1_coupling_H.irow)
    construct_intra_d_coup(intra_d1_coupling_num, self.d1_coupling_dmat_index, self.detector1.d_H.mat, self.d1_H)

    # coupling in detector2
    intra_d2_coupling_num = len(self.d2_coupling_H.irow)
    construct_intra_d_coup(intra_d2_coupling_num, self.d2_coupling_dmat_index, self.detector2.d_H.dmat, self.d2_H)



