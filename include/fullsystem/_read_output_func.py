import numpy as np
import os

# ------------ read & output function ---------------------------------------
# -------------------------- Read off-diag parameter num  ---------------------
def read_offdiag_coupling_element(self, offdiagonal_coupling_list):
    '''

    :param offdiagonal_coupling_list: python list. off-diag coupling param.
    :return:
    '''
    self.offdiag_param_list = offdiagonal_coupling_list.copy()

    # intra-detector coupling matrix element in detector 1.
    begin_index = 0
    end_index = self.detector1.show_offdiag_matrix_num()
    # always deep-copy the array to ensure encapsulation of the data.
    d1_offdiag_param = offdiagonal_coupling_list [begin_index: end_index].copy()
    # construct anharmonic (off-diagonal) coupling in detector 1.
    self.detector1.construct_offdiag_coupling_value(d1_offdiag_param)

    # intra-detector coupling matrix element in detector 2.
    begin_index = end_index
    end_index = end_index + self.detector2.show_offdiag_matrix_num()
    # always deep-copy the array to ensure encapsulation of the data.
    d2_offdiag_param = offdiagonal_coupling_list [begin_index : end_index].copy()
    # construct anharmonic (off-diagonal) coupling in detector 2
    self.detector2.construct_offdiag_coupling_value(d2_offdiag_param)

    # photon-detector coupling matrix element.
    begin_index = end_index
    end_index = begin_index + self.pd_coupling_num
    self.pd_offdiag_param = offdiagonal_coupling_list[begin_index : end_index].copy()

    # detector-detector coupling matrix element
    begin_index = end_index
    end_index = begin_index + self.dd_coupling_num
    self.dd_offdiag_param = offdiagonal_coupling_list[begin_index : end_index].copy()


def output_state_qn_number_list(self):
    # output quantum num for state
    print(self._basis_set_state_qn_list)


def output_off_diagonal_coupling_state_pairs_info(self):
    coupling_mode_list = []
    for i in range(self._basis_set_state_num, self._mat_num , 2):
        irow_index = self._irow[i]
        icol_index = self._icol[i]

        coupling_state_qn = []
        coupling_state_qn.append(self._basis_set_state_qn_list[irow_index])
        coupling_state_qn.append(self._basis_set_state_qn_list[icol_index])

        coupling_mode_list.append(coupling_state_qn)

    list_len = len(coupling_mode_list)
    print("Coupling number:  "  + str(list_len) + "\n")
    print("Coupling for state in full system: ")
    for i in range(list_len):
        print(coupling_mode_list[i])