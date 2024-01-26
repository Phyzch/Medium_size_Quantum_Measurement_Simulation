import numpy as np
from include.detector_class.__init__ import Detector
from include.Constructing_state_module import binary_search_qn_list

'''
first part of construcing Hamiltonian. This only have to be called once.
We do following : 1. compute energy of full_system  : self.__compute_initial_energy()
                  2. let detector construct Hamiltonian and coupling info: self.detector1/2.construct_detector_Hamiltonian_part1()
                  3. construct diagonal part of Hamiltonian and state for full_system : self.__construct_full_system_diagonal_Hamiltonian()
                  4. construct irow, icol for off-diagonal coupling between state : self.__construct_offdiag_dd_pd_coup() , self.__construct_intra_detector_coupling()
'''

def construct_full_system_Hamiltonian_part1(self):
    self._compute_initial_energy()

    self.detector1.construct_detector_Hamiltonian_part1()
    self.detector2.construct_detector_Hamiltonian_part1()

    self._construct_full_system_diagonal_Hamiltonian()

    # compute  parameter number and irow, icol for coupling between 2 detector and coupling beteween photon and detector.
    self._construct_offdiag_dd_pd_coup()

    # compute position of intra-detector coupling
    self._construct_intra_detector_coupling()


def _compute_initial_energy(self):
    self.initial_energy = 0
    d1_energy = np.sum(np.array(self.detector1._frequency) * np.array(self.detector1._initial_state_qn))
    d2_energy = np.sum(np.array(self.detector2._frequency) * np.array(self.detector2._initial_state_qn))

    self.initial_energy = self.init_photon_energy + d1_energy + d2_energy


def _construct_full_system_diagonal_Hamiltonian(self):
    '''
    construct state and diagonal part of Hamiltonian.
    impose energy window : states included should satisfy : |E - E_init | <= energy_window
    :return:
    '''
    self._basis_set_state_num = 0
    for i in range(self.photon_state_num):
        for j in range(self.detector1._basis_set_state_num):
            for k in range(self.detector2._basis_set_state_num):
                energy = self.photon_state_energy[i] + self.detector1._basis_set_state_energy_list[j] + \
                         self.detector2._basis_set_state_energy_list[k]

                if (abs(energy - self.initial_energy) <= self.energy_window):
                    self.sstate.append(i)
                    self.dstate1.append(j)
                    self.dstate2.append(k)

                    state_mode = [self.photon_state_mode[i], self.detector1._basis_set_state_qn_list[j].tolist(),
                                  self.detector2._basis_set_state_qn_list[k].tolist()]
                    self._basis_set_state_qn_list.append(state_mode)

                    # Hamiltonian for H
                    self.full_H.append(energy, self._basis_set_state_num, self._basis_set_state_num)

                    # Hamiltonian for photon, d1, d2
                    self.photon_H.append(self.photon_state_energy[i], self._basis_set_state_num, self._basis_set_state_num)
                    self.d1_H.append(self.detector1._basis_set_state_energy_list[j], self._basis_set_state_num, self._basis_set_state_num)
                    self.d2_H.append(self.detector2._basis_set_state_energy_list[k], self._basis_set_state_num, self._basis_set_state_num)

                    self._basis_set_state_num = self._basis_set_state_num + 1

    # shift Hamiltonian's diag part by initial energy to speed up propagation of wave function.
    self._Shift_Hamiltonian()

    # diagonal part of Hamiltonian. No coupling.
    self.full_H._diagonal_mat = self.full_H._mat.copy()
    self.d1_H._diagonal_mat = self.d1_H._mat.copy()
    self.d2_H._diagonal_mat = self.d2_H._mat.copy()

def _Shift_Hamiltonian(self):
    '''
    shift Hamiltonian by energy : <\psi | H | \psi>
    '''
    for i in range(self._basis_set_state_num):
        self.full_H._mat[i] = self.full_H._mat[i] - self.initial_energy

def _construct_intra_detector_coupling(self):
    # -------------- inline function -------
    def construct_intra_d_coupling_submodule (i, j, di, dj, dstate_num, dmat_num, dirow, dicol,
                                   d_coupling_H, d_coupling_dmat_index,
                                   dmat_H):
        '''
        :param i, j: state index in full_system
        :param di, dj : state index in detector .
        :param dstate_num: number of state in detector Hamiltonian
        :param dmat_num:  number of matrix term in detector Hamiltonian
        :param dirow , dicol : row and column for detector matrix
        :param d_coupling_H : detector coupling in full system's Hamiltonian
        :param d_coupling_dmat_index: index of d matrix in full system's Hamiltonian
        :param dmat_H : H_d \otimes I_d \otimes I_sys.  detector Hamiltonian in representation of full_system Hamiltonian.
        :return:
        '''

        for k in range(dstate_num, dmat_num):
            # k is off-diagonal matrix index in detector's Hamiltonian
            if (dirow[k] == di and dicol[k] == dj):
                # d_coupling only record intra-detector coupling matrix index
                d_coupling_H._irow.append(i)
                d_coupling_H._icol.append(j)
                d_coupling_dmat_index.append(k)

                # d_irwo ,  dicol
                dmat_H._irow.append(i)
                dmat_H._icol.append(j)

                dmat_H._irow.append(j)
                dmat_H._icol.append(i)
                break

    # --------------- inline function -------------------------------------

    for i in range(self._basis_set_state_num):
        for j in range(i + 1, self._basis_set_state_num):
            ss = self.sstate[i] - self.sstate[j]

            # i , j stands for different state.  1, 2 stand for detector 1 and detector 2
            di1 = self.dstate1[i]
            di2 = self.dstate2[i]
            dj1 = self.dstate1[j]
            dj2 = self.dstate2[j]

            # coupling in detector2
            if (ss == 0 and di1 == dj1 and di2 != dj2):
                construct_intra_d_coupling_submodule(i, j, di2, dj2, self.detector2._basis_set_state_num, self.detector2._mat_num,
                                                     self.detector2._d_Hamiltonian._irow, self.detector2._d_Hamiltonian._icol, self.d2_coupling_H,
                                                     self.d2_coupling_dmat_index,
                                                     self.d2_H)

            # coupling in detector 1
            elif (ss == 0 and di1 != dj1 and di2 == dj2):
                construct_intra_d_coupling_submodule(i, j, di1, dj1, self.detector1._basis_set_state_num, self.detector1._mat_num,
                                                     self.detector1._d_Hamiltonian._irow, self.detector1._d_Hamiltonian._icol, self.d1_coupling_H,
                                                     self.d1_coupling_dmat_index,
                                                     self.d1_H)

    #  construct irow and icol. (Note for irow,icol. We add off diagonal part between detector in compute_full_system_offdiagonal_paramter_number())
    #  Then we add offdiagonal index within same detector below. Same order apply to part that reconstruct offdiagonal part of mat.
    _insert_upper_lower_triangular_index(recv_H=self.full_H, input_H=self.d1_coupling_H)
    _insert_upper_lower_triangular_index(recv_H=self.full_H, input_H=self.d2_coupling_H)


def _construct_offdiag_dd_pd_coup(self):
    '''

    :return:
    '''

    # ----------- inline function ---------------
    def include_pd_coupling(d_dof, state_mode_list, di, dj, pd_coupling_num):
        '''

        :param d_dof: detector dof
        :param state_mode_list: mode_list for d
        :param di: row index in d matrix
        :param dj: column index in d matrix
        :return:
        '''
        if(state_mode_list[di][0] - state_mode_list[dj][0] == 1 ):
                same = True
                for k in range(1, d_dof):
                    if (state_mode_list[di][k] != state_mode_list[dj][k]):
                        same = False
                        break

                if (same):
                    # include this coupling in matrix and irow, icol.
                    self.offdiag_param_num = self.offdiag_param_num + 1
                    # As irow, icol for Hamiltonian will not change during Genetic algorithm (only value of coupling will change, we construct irow , icol here)
                    self.full_H._irow.append(i)
                    self.full_H._icol.append(j)
                    # lower triangular part.
                    self.full_H._irow.append(j)
                    self.full_H._icol.append(i)

                    self.pd_dd_coupling_irow.append(i)
                    self.pd_dd_coupling_icol.append(j)

                    pd_coupling_num = pd_coupling_num + 1

        return pd_coupling_num

    def examine_dd_coupling(d_dof, state_mode_list, di, dj):
        succeed = False
        for k in range(d_dof):
            deldv = state_mode_list[di][k] - state_mode_list[dj][k]
            if (abs(deldv) == 1):
                zero = 0
                zero = zero + np.sum(np.abs(state_mode_list[di][:k] - state_mode_list[dj][:k]))
                zero = zero + np.sum(np.abs(state_mode_list[di][k + 1:] - state_mode_list[dj][k + 1:]))

                if (zero == 0):
                    succeed = True
                    return succeed
                else:
                    succeed = False
                    return succeed

        return succeed

    def include_dd_coupling(di1, dj1, di2, dj2, dd_coupling_num):
        '''
        include coupling between detectors
        :return:
        '''
        d1_succeed = examine_dd_coupling(self.detector1._dof, self.detector1._basis_set_state_qn_list, di1, dj1)
        d2_succeed = examine_dd_coupling(self.detector2._dof, self.detector2._basis_set_state_qn_list, di2, dj2)

        if d1_succeed and d2_succeed:
            self.offdiag_param_num = self.offdiag_param_num + 1
            self.pd_dd_coupling_irow.append(i)
            self.pd_dd_coupling_icol.append(j)

            self.full_H._irow.append(i)
            self.full_H._icol.append(j)
            # lower triangular part.
            self.full_H._irow.append(j)
            self.full_H._icol.append(i)

            dd_coupling_num = dd_coupling_num + 1

        return dd_coupling_num

    # --------- inline function --------------

    self.offdiag_param_num = self.offdiag_param_num + self.detector1._offdiagonal_coupling_num
    self.offdiag_param_num = self.offdiag_param_num + self.detector2._offdiagonal_coupling_num

    pd_coupling_num = 0
    dd_coupling_num = 0

    # count coupling between system and detector
    for i in range(self._basis_set_state_num):
        for j in range(i + 1, self._basis_set_state_num):
            ss = self.sstate[i] - self.sstate[j]

            di1 = self.dstate1[i]
            di2 = self.dstate2[i]
            dj1 = self.dstate1[j]
            dj2 = self.dstate2[j]


            # no coupling between photon state [0,1] & [1,0]
            if (self.sstate[i] + self.sstate[j] == 3):
                ss = -3

            # coupling for photon with detector1
            # photon state: [0,0] and [1,0]
            if (ss == -1 and di1 != dj1 and di2 == dj2):
                pd_coupling_num = include_pd_coupling(self.detector1._dof, self.detector1._basis_set_state_qn_list, di1, dj1,
                                                      pd_coupling_num)

            # coupling for photon with detector2
            if (ss == -2 and di1 == dj1 and di2 != dj2):
                pd_coupling_num = include_pd_coupling(self.detector2._dof, self.detector2._basis_set_state_qn_list, di2, dj2,
                                                      pd_coupling_num)

            # coupling between detector1 and detector2
            if (ss == 0 and di1 != dj1 and di2 != dj2):
                dd_coupling_num = include_dd_coupling(di1, dj1, di2, dj2, dd_coupling_num)

    self.pd_coupling_num = pd_coupling_num
    self.dd_coupling_num = dd_coupling_num

def _insert_upper_lower_triangular_index(recv_H, input_H):
    '''

    :param recv_H:  Hamiltonian to recv input_H 's irow, icol. irow, icol will duplicate element in input_H as upper and lower triangular matrix index
    :param input_H: Hamiltnian for input.
    :return:
    '''
    num = len(input_H._irow)
    for i in range(num):
        # upper diagonal part
        recv_H._irow.append(input_H._irow[i])
        recv_H._icol.append(input_H._icol[i])
        # lower diagonal part
        recv_H._irow.append(input_H._icol[i])
        recv_H._icol.append(input_H._irow[i])
