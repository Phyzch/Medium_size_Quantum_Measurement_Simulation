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

    begin_index = 0
    end_index = self.detector1.offdiag_coupling_num
    d1_offdiag_param = offdiagonal_coupling_list [ begin_index: end_index].copy()

    begin_index = end_index
    end_index = end_index + self.detector2.offdiag_coupling_num
    d2_offdiag_param = offdiagonal_coupling_list [ begin_index : end_index ].copy()

    begin_index = end_index
    end_index = self.offdiag_param_num
    self.pd_dd_offdiag_param = offdiagonal_coupling_list[begin_index: end_index].copy()

    return d1_offdiag_param, d2_offdiag_param

def output_state_mode(self):
    # output quantum num for state
    print(self.state_mode_list)

# ------ output offdiagonal parameter number --------
def output_offdiagonal_parameter_number(self):
    # we need to output offdiagonal parameter number to tell Genetic algorithm how many parameters we need to sample
    return self.offdiag_param_num

def output_off_diagonal_coupling_mode_info(self):
    Coupling_mode_list = []
    self.state_num = self.full_H.statenum()
    self.matnum = self.full_H.matnum()
    for i in range(self.state_num, self.matnum , 2):
        irow_index = self.full_H.irow[i]
        icol_index = self.full_H.icol[i]

        coupling_mode = []
        coupling_mode.append(self.state_mode_list[irow_index])
        coupling_mode.append(self.state_mode_list[icol_index])

        Coupling_mode_list.append(coupling_mode)

    list_len = len(Coupling_mode_list)
    print("Coupling number:  "  + str(list_len) + "\n")
    print("Coupling for state in full system: ")
    for i in range(list_len):
        print(Coupling_mode_list[i])