import numpy as np
import Shared_data
from Detector_class import detector
from Constructing_state_module import binary_search_mode_list

import numba
from numba import jit

'''
This class serve as creating quantum system which consist of 
two medium-size-subsystem (we call it detector) and one small-size subsystem (we call it photon)
The coupling parameter between states is optimized using Genetic algorithm if we want to search best parameter to give best localization.
'''

# To use this class:
# fixme: First run:  construct_full_system_Hamiltonian_part1
# fixme: Then run: output_offdiagonal_parameter_number(self) to tell Genetic algorithm number of off-diagonal parameter we need to feed
# fixme: Then run construct_full_system_Hamiltonian_part2(self , offdiagonal_coupling_list) [This is called in Genetic algorithm fitness function]

@jit(nopython = True)
def wave_func_sum(original_value, part_add, index):
    '''
    H_real[ irow[index]] = H_real[irow[index]] + part_add[index]
    :param original_value:
    :param part_add:
    :param index:
    :return:
    '''
    Len = np.shape(index)[0]
    for i in range(Len):
        index_i = index[i]
        original_value[index_i] = original_value[index_i] + part_add[i]

    return original_value

class Hamiltonian():
    # a primitive class for Hamiltonian:
    # irow : row index for Hamiltonian
    irow = []
    # icol: column index for Hamiltonian
    icol = []
    # mat: matrix value for Hamiltonian.
    mat = []

class full_system():

    def __init__(self, Detector_1_parameter, Detector_2_parameter, energy_window, photon_energy, initial_photon_wave_function):
        # energy window for full matrix is passed through parameter directly here.

        # we have three energy windows. energy_window for each detector is contained in detector_parameter.
        self.detector1 = detector(Detector_1_parameter)
        self.detector2 = detector(Detector_2_parameter )

        # initial wave function for photon
        self.initial_photon_wave_function = initial_photon_wave_function
        # initial energy of photon + detector 1 + detector 2
        self.initial_energy = 0
        
        # for every state in full system, it composed of system state, detector 1 's state, detector 2's state.
        # state_mode_list contain information about : [ photon_mode, detector1_mode, detector2_mode ]
        self.sstate = []
        self.dstate1 = []
        self.dstate2 = []
        self.state_mode_list = []


        # system state is [00, 01, 10], thus system's state number is 3.
        self.photon_state_num = 3
        # energy of photon
        self.photon_energy = photon_energy
        self.photon_state_energy = [0, photon_energy, photon_energy]
        self.photon_state_mode = [[0, 0], [1, 0], [0, 1]]

        # energy window is used to construct state in full system relative to initial state.
        self.energy_window = energy_window
        self.state_num = 0
        self.matnum = 0

        # offdiagonal_parameter_number is output to Genetic algorithm for optimization.
        # off_diagonal_parameter in list is in order: intra-detector coupling in detector 1, intra-detector couping in detector 2. coupling between detector and photon , coupling between detector and deteoctr.
        self.offdiag_param_num = 0
        # off diagonal parameter : 1. parameter for intra-detector coupling  2. coupling between detectors.
        self.offdiag_param_list = []
        # offdiagonal parameter : 1. between system and detector and 2.between detector and detector.
        self.offdiag_param_list_inter = []

        # mat: Hamiltonian (sparse matrix). irow, icol : row, col index.
        self.mat = []
        self.irow = []
        self.icol = []
        self.mat_diagonal_part = []

        # off diagonal coupling element in detector1 and detector2 , their corresponding index in full system matrix :
        self.d1_coupling_irow = []
        self.d1_coupling_icol = []
        self.d1_coupling_dmat_index = []

        self.d2_coupling_irow = []
        self.d2_coupling_icol = []
        self.d2_coupling_dmat_index = []

        # coupling index between detector and system. also between detector and detector.
        self.pd_dd_coupling_irow = []
        self.pd_dd_coupling_icol = []


        # wave function:
        self.photon_wave_func = np.zeros(self.photon_state_num)
        self.photon_wave_func[1] = 1/np.sqrt(2)
        self.photon_wave_func[2] = 1/np.sqrt(2)
        self.wave_function = []

        # Hamiltonian solely for photon in full matrix:
        self.photon_mat = []
        self.photon_irow = []
        self.photon_icol = []

        # Hamiltonian solely for detector1 in full matrix
        self.d1_mat = []
        self.d1_irow = []
        self.d1_icol = []

        # Hamiltonian solely for detectro2 in full matrix
        self.d2_mat = []
        self.d2_irow = []
        self.d2_icol = []

        # diagonal part for detector1 and detector2 's Hamiltonian
        self.d1_mat_diagonal = []
        self.d2_mat_diagonal = []

        # sort state by layer . Record their index and state mode
        self.state_mode_by_layer = []
        self.state_index_by_layer = []

# ----------------   first part of construcing Hamiltonian. This only have to be called once. --------------------------------
    def compute_initial_energy(self):
        self.initial_energy = 0
        d1_energy = np.sum(np.array(self.detector1.frequency ) * np.array(self.detector1.initial_state))
        d2_energy = np.sum(np.array(self.detector2.frequency)  * np.array( self.detector2.initial_state))

        self.initial_energy = self.photon_energy + d1_energy + d2_energy

    def construct_full_system_Hamiltonian_part1(self):
        self.compute_initial_energy()

        self.detector1.construct_detector_Hamiltonian_part1()
        self.detector2.construct_detector_Hamiltonian_part1()

        self.construct_full_system_diagonal_Hamiltonian()

        # fixme: sort state by layer. In construction.
        # Code to write sort index according to their layer. Layer is determined by number of nonzero modes. (2 mode , 3 mode , 4 mode etc).
        # self.Sort_state_by_layer()

        # compute offdiagonal parameter number and record irow, icol for coupling.
        self.construct_offdiag_param_coup()

        # compute position of intra-detector coupling
        self.compute_position_of_intra_detector_coupling()

        # fixme: decide layer the coupling parameter belong to:


    def construct_full_system_diagonal_Hamiltonian(self):
        '''
        construct state and diagonal part of Hamiltonian.
        impose energy window : states included should satisfy : |E - E_init | <= energy_window
        :return:
        '''
        self.state_num = 0
        for i in range(self.photon_state_num):
            for j in range(self.detector1.state_num):
                for k in range(self.detector2.state_num):
                    energy = self.photon_state_energy[i] + self.detector1.State_energy_list[j] + \
                             self.detector2.State_energy_list[k]

                    if( abs(energy - self.initial_energy) <= self.energy_window ):
                        self.sstate.append(i)
                        self.dstate1.append(j)
                        self.dstate2.append(k)

                        state_mode =[]
                        state_mode.append(self.photon_state_mode[i])
                        state_mode.append(self.detector1.state_mode_list[j].tolist())
                        state_mode.append(self.detector2.state_mode_list[k].tolist())
                        self.state_mode_list.append(state_mode)

                        self.mat.append( energy )
                        self.irow.append(self.state_num)
                        self.icol.append(self.state_num)
                        self.state_num = self.state_num + 1

                        # Hamiltonian for photon, d1, d2
                        self.photon_mat.append(self.photon_state_energy[i])
                        self.d1_mat.append(self.detector1.State_energy_list[j])
                        self.d2_mat.append(self.detector2.State_energy_list[k])

        # diagonal part of Hamiltonian. No coupling.
        self.mat_diagonal_part = self.mat.copy()

        self.photon_irow = self.irow.copy()
        self.photon_icol = self.icol.copy()

        self.d1_irow = self.irow.copy()
        self.d1_icol = self.icol.copy()

        self.d2_irow = self.irow.copy()
        self.d2_icol = self.icol.copy()

        self.d1_mat_diagonal = self.d1_mat.copy()
        self.d2_mat_diagonal = self.d2_mat.copy()

    def Sort_state_by_layer(self):
        dof = self.detector1.dof
        for layer_dof in range(2, dof+1):
            index_list = []
            mode_number_list = []
            for i in range(self.state_num):
                state_mode = self.state_mode_list[i]
                detector1_state_mode = state_mode[1]
                detector2_state_mode = state_mode[2]
                if(layer_dof == 2):
                    # we include mode 0 and mode 1 state. Thus we do not check mode 2 is nonzero but only require mode 3 ~ higher order is 0
                    Zero_bool = True
                    for j in range(layer_dof, dof):
                        if(detector1_state_mode[j] != 0):
                            Zero_bool = False
                            break
                        if(detector2_state_mode[j] != 0):
                            Zero_bool = False
                            break

                    if(Zero_bool == True):
                        index_list.append(i)
                        mode_number_list.append(state_mode)

                else:
                    if(detector1_state_mode[layer_dof - 1] != 0 and detector2_state_mode[layer_dof - 1] !=0 ):
                        Zero_bool = True
                        for j in range(layer_dof, dof):
                            if(detector1_state_mode[j]!= 0):
                                Zero_bool = False
                                break
                            if(detector2_state_mode[j]!=0):
                                Zero_bool = False
                                break

                        if(Zero_bool == True):
                            index_list.append(i)
                            mode_number_list.append(state_mode)

            self.state_mode_by_layer.append(mode_number_list)
            self.state_index_by_layer.append(index_list)

    def Sort_coupling_by_layer(self):
        '''

        :return:
        '''


    def compute_position_of_intra_detector_coupling(self):

        def construct_intra_d_coupling( i, j,  di, dj , dstate_num , dmat_num , dirow, dicol,
                                       d_coupling_irow, d_coupling_icol , d_coupling_dmat_index,
                                       dmat_irow, dmat_icol):
            '''
            :param i, j: state index in full_system
            :param di, dj : state index in detector .
            :param dstate_num: number of state in detector Hamiltonian
            :param dmat_num:  number of matrix term in detector Hamiltonian
            :param dirow , dicol : row and column for detector matrix
            :param d_coupling_irow , d_coupling_icol : row and column index for detector coupling in full system's Hamiltonian
            :param d_coupling_dmat_index: index of d matrix in full system's Hamiltonian
            :param dmat_irow , dmat_icol : H_d \otimes I_d \otimes I_sys.  detector Hamiltonian in representation of full_system Hamiltonian.
            this is d_diagonal + d_coupling
            :return:
            '''

            for k in range(dstate_num , dmat_num):
                # k is off-diagonal matrix index in detector's Hamiltonian
                if(dirow[k] == di and dicol[k] == dj ):
                    # d_coupling only record intra-detector coupling matrix index
                    d_coupling_irow.append(i)
                    d_coupling_icol.append(j)
                    d_coupling_dmat_index.append(k)

                    # d_irwo ,  dicol
                    dmat_irow.append(i)
                    dmat_icol.append(j)

                    dmat_irow.append(j)
                    dmat_icol.append(i)
                    break


        for i in range(self.state_num):
            for j in range(i + 1, self.state_num):
                ss = self.sstate[i] - self.sstate[j]

                # i , j stands for different state.  1, 2 stand for detector 1 and detector 2
                di1 = self.dstate1[i]
                di2 = self.dstate2[i]
                dj1 = self.dstate1[j]
                dj2 = self.dstate2[j]

                # coupling in detector2
                if (ss == 0 and di1 == dj1 and di2 != dj2):
                    construct_intra_d_coupling(i,j, di2, dj2 , self.detector2.state_num , self.detector2.dmatnum ,
                                               self.detector2.dirow, self.detector2.dicol, self.d2_coupling_irow,
                                               self.d2_coupling_icol, self.d2_coupling_dmat_index,
                                               self.d2_irow, self.d2_icol)

                # coupling in detector 1
                elif (ss == 0 and di1 != dj1 and di2 == dj2):
                    construct_intra_d_coupling(i,j,di1,dj1, self.detector1.state_num , self.detector1.dmatnum,
                                               self.detector1.dirow, self.detector1.dicol, self.d1_coupling_irow,
                                               self.d1_coupling_icol, self.d1_coupling_dmat_index,
                                               self.d1_irow, self.d1_icol)

        #  construct irow and icol. (Note for irow,icol. We add off diagonal part between detector in compute_full_system_offdiagonal_paramter_number())
        #  Then we add offdiagonal index within same detector below. Same order apply to part that reconstruct offdiagonal part of mat.
        intra_detector1_coupling_num = len(self.d1_coupling_irow)
        for i in range(intra_detector1_coupling_num):
            self.irow.append(self.d1_coupling_irow[i])
            self.icol.append(self.d1_coupling_icol[i])

            #lower diangular part
            self.irow.append(self.d1_coupling_icol[i])
            self.icol.append(self.d1_coupling_irow[i])

        intra_detector2_coupling_num = len(self.d2_coupling_irow)
        for i in range(intra_detector2_coupling_num):
            self.irow.append(self.d2_coupling_irow[i])
            self.icol.append(self.d2_coupling_icol[i])

            # lower diagonal part
            self.irow.append(self.d2_coupling_icol[i])
            self.icol.append(self.d2_coupling_irow[i])

        self.matnum = len(self.irow)


    def construct_offdiag_param_coup(self):
        '''

        :return:
        '''
        def include_pd_coupling(d_dof, state_mode_list , di, dj , pd_coupling_num):
            '''

            :param d_dof: detector dof
            :param state_mode_list: mode_list for d
            :param di: row index in d matrix
            :param dj: column index in d matrix
            :return:
            '''
            if(state_mode_list[di][0] == state_mode_list[dj][0] ):
                same = True
                for k in range(1, d_dof):
                    if(state_mode_list[di][k] != state_mode_list[dj][k] ):
                        same = False
                        break

                if(same):
                    # include this coupling in matrix and irow, icol.
                    self.offdiag_param_num = self.offdiag_param_num + 1
                    # As irow, icol for Hamiltonian will not change during Genetic algorithm (only value of coupling will change, we construct irow , icol here)
                    self.irow.append(i)
                    self.icol.append(j)
                    # lower triangular part.
                    self.irow.append(j)
                    self.icol.append(i)

                    self.pd_dd_coupling_irow.append(i)
                    self.pd_dd_coupling_icol.append(j)

                    pd_coupling_num = pd_coupling_num + 1

            return pd_coupling_num

        def examine_dd_coupling(d_dof , state_mode_list, di, dj):
            succeed = False
            for k in range(d_dof):
                deldv = state_mode_list[di][k] - state_mode_list[dj][k]
                if(abs(deldv) == 1):
                    zero = 0
                    zero = zero + np.sum ( np.abs( state_mode_list[di][:k] - state_mode_list[dj][:k] ) )
                    zero = zero + np.sum(  np.abs( state_mode_list[di][k+1 : ] - state_mode_list[dj][k+1 : ] ))

                    if(zero == 0):
                        succeed = True
                        return succeed
                    else:
                        succeed = False
                        return succeed

            return succeed

        def include_dd_coupling(di1, dj1, di2, dj2 , dd_coupling_num):
            '''
            include coupling between detectors
            :return:
            '''
            d1_succeed = examine_dd_coupling(self.detector1.dof, self.detector1.state_mode_list, di1, dj1)
            d2_succeed = examine_dd_coupling(self.detector2.dof, self.detector2.state_mode_list, di2, dj2)

            if d1_succeed and d2_succeed :
                self.offdiag_param_num = self.offdiag_param_num + 1
                self.pd_dd_coupling_irow.append(i)
                self.pd_dd_coupling_icol.append(j)

                self.irow.append(i)
                self.icol.append(j)
                # lower triangular part.
                self.irow.append(j)
                self.icol.append(i)

                dd_coupling_num = dd_coupling_num + 1

            return dd_coupling_num

        self.offdiag_param_num = self.offdiag_param_num + self.detector1.offdiag_coupling_num
        self.offdiag_param_num = self.offdiag_param_num + self.detector2.offdiag_coupling_num

        pd_coupling_num = 0
        dd_coupling_num = 0

        # count coupling between system and detector
        for i in range(self.state_num):
            for j in range( i+1 , self.state_num):
                ss = self.sstate[i] - self.sstate[j]

                di1 = self.dstate1[i]
                di2 = self.dstate2[i]
                dj1 = self.dstate1[j]
                dj2 = self.dstate2[j]

                # no coupling between photon state [0,1] & [1,0]
                if(self.sstate[i] + self.sstate[j] == 3):
                    ss= -3

                # coupling for photon with detector1
                # photon state: [0,0] and [1,0]
                if(ss == -1 and di1 != dj1 and di2 == dj2):
                    pd_coupling_num = include_pd_coupling(self.detector1.dof, self.detector1.state_mode_list, di1, dj1, pd_coupling_num)

                # coupling for photon with detector2
                if(ss == -2 and di1 == dj1 and di2 != dj2):
                    pd_coupling_num = include_pd_coupling(self.detector2.dof, self.detector2.state_mode_list, di2, dj2, pd_coupling_num)

                # coupling between detector1 and detector2
                if(ss ==0 and di1!= dj1 and di2 != dj2):
                    dd_coupling_num = include_dd_coupling(di1,dj1,di2,dj2,dd_coupling_num)



    def print_state_mode(self):
        # output function.
        print(self.state_mode_list)

#  --------------------------------------- first part of construcing Hamiltonian.  End ---------------------

 # -------------------------- Read and output offdiagonal parameter number . Also reverse matrix begin ---------------------
    def output_offdiagonal_parameter_number(self):
        # we need to output offdiagonal parameter number to tell Genetic algorithm how many parameters we need to sample
        return self.offdiag_param_num

    def read_offdiag_coupling_element(self,offdiagonal_coupling_list):
        self.offdiag_param_list = offdiagonal_coupling_list.copy()

        begin_index = 0
        end_index = self.detector1.offdiag_coupling_num
        off_diagonal_parameter_for_detector1 = offdiagonal_coupling_list [ begin_index: end_index].copy()

        begin_index = begin_index + self.detector1.offdiag_coupling_num
        end_index = end_index + self.detector2.offdiag_coupling_num
        off_diagonal_parameter_for_detector2 = offdiagonal_coupling_list [ begin_index : end_index ].copy()

        begin_index = begin_index + self.detector2.offdiag_coupling_num
        end_index = self.offdiag_param_num
        self.offdiag_param_list_inter = offdiagonal_coupling_list[begin_index: end_index].copy()

        # each detector construct their hamiltonian
        self.detector1.construct_detector_Hamiltonian_part2( off_diagonal_parameter_for_detector1 )
        self.detector2.construct_detector_Hamiltonian_part2( off_diagonal_parameter_for_detector2 )

    def Reverse_mat(self):
        # For each generation, we only have to update off-diagonal part .
        self.detector1.Reverse_dmat()
        self.detector2.Reverse_dmat()

        self.mat = self.mat_diagonal_part.copy()

        self.d1_mat = self.d1_mat_diagonal.copy()
        self.d2_mat = self.d2_mat_diagonal.copy()


    # -------------------------- Read and output offdiagonal parameter number . Also reverse matrix  End---------------------

    def construct_full_system_offdiag_coupling(self):

        inter_detector_coupling_num = len(self.pd_dd_coupling_irow)
        if(inter_detector_coupling_num != len(self.offdiag_param_list_inter)):
            raise NameError("inter detector coupling number does not equal to parameter number read from Genetic algorithm")

        # coupling between detector and system. and detector between detector
        for i in range(inter_detector_coupling_num):
            self.mat.append(self.offdiag_param_list_inter[i])

            self.mat.append(self.offdiag_param_list_inter[i])

        # coupling in detector 1
        intra_detector1_coupling_num = len(self.d1_coupling_irow)
        for i in range(intra_detector1_coupling_num):
            k = self.d1_coupling_dmat_index[i]

            self.mat.append(self.detector1.dmat[ k ])

            # we also record lower trangular part
            self.mat.append(self.detector1.dmat[ k ])


            # We construct Hamiltonian for detector1
            self.d1_mat.append(self.detector1.dmat[ k])
            # also lower triangular part
            self.d1_mat.append(self.detector1.dmat[ k])


        # coupling in detector2
        intra_detector2_coupling_num = len(self.d2_coupling_irow)
        for i in range(intra_detector2_coupling_num):
            k = self.d2_coupling_dmat_index[i]

            self.mat.append(self.detector2.dmat[ k ])

            self.mat.append(self.detector2.dmat[ k ])

            # We construct Hamiltonian for detector2
            self.d2_mat.append(self.detector2.dmat[ k])
            # also lower triangular part
            self.d2_mat.append(self.detector2.dmat[ k])



    def construct_full_system_Hamiltonian_part2(self , offdiagonal_coupling_list):
        '''
        After we read offdiagonal parameter from Genetic algorithm, we do this part.
        offdiagonal_coupling_list : size [self.offdiagonal coupling num]
        :return:
        '''
        # First reverse matrix to contain only diagonal part.
        self.Reverse_mat()

        # Then read offdiagonal coupling parameter
        self.read_offdiag_coupling_element(offdiagonal_coupling_list)

        # full system construct Hamiltonian using detector's Hamiltonian.
        self.construct_full_system_offdiag_coupling()

        # initialize wave function.
        self.initialize_wave_function()

        # shift Hamiltonian
        self.Shift_Hamiltonian()


    def initialize_wave_function(self):
        self.detector1.initialize_wave_function()
        self.detector2.initialize_wave_function()

        # position1 and position2 is postion in detector matrix.
        position1, exist1 = binary_search_mode_list(self.detector1.state_mode_list, self.detector1.initial_state)
        position2, exist2 = binary_search_mode_list(self.detector2.state_mode_list, self.detector2.initial_state)

        self.wave_function = np.zeros(self.state_num, dtype = np.complex)

        for i in range(self.state_num):
            if self.sstate[i] == 1  :
                if self.dstate1[i] == position1 and self.dstate2[i] == position2 :
                    self.wave_function[i] = self.initial_photon_wave_function[0]

            if self.sstate[i] == 2 :
                if self.dstate1[i] == position1 and self.dstate2[i] == position2 :
                    self.wave_function[i] = self. initial_photon_wave_function[1]

    def Shift_Hamiltonian(self):
        for i in range(self.state_num):
            self.mat[i] = self.mat[i] - self.initial_energy


    def Evaluate_photon_energy(self):
        # use self.mat_photon and self.photon_irow. self.photon_icol
        H_phi = self.photon_mat * self.wave_function[self.photon_icol]

        H_phi_wave_function = np.zeros(self.state_num,dtype=np.complex)
        H_phi_wave_function = wave_func_sum(H_phi_wave_function,H_phi, self.photon_irow)

        photon_energy = np.sum (np.real(np.conjugate(self.wave_function) * H_phi_wave_function) )

        return photon_energy

    def Evaluate_detector1_energy(self):
        H_phi = self.d1_mat * self.wave_function[self.d1_icol]

        H_phi_wave_function = np.zeros(self.state_num,dtype=np.complex)
        H_phi_wave_function = wave_func_sum(H_phi_wave_function, H_phi, self.d1_irow)

        detector1_energy = np.sum( np.real( np.conjugate(self.wave_function) * H_phi_wave_function ))

        return detector1_energy

    def Evaluate_detector2_energy(self):
        H_phi = self.d2_mat * self.wave_function[self.d2_icol]

        H_phi_wave_function = np.zeros(self.state_num,dtype=np.complex)
        H_phi_wave_function = wave_func_sum(H_phi_wave_function, H_phi, self.d2_irow)

        detector2_energy = np.sum(np.real(np.conjugate(self.wave_function) * H_phi_wave_function))

        return detector2_energy

    def Evolve_dynamics(self):
        Final_time = Shared_data.Time_duration
        output_time_step = Shared_data.output_time_step

        # define time step to do simulation
        Max_element = np.max( np.abs(self.mat) )
        time_step = 0.2 / (Max_element)

        # output step number and total_step_number
        output_step_number = max( int(output_time_step / time_step) , 1)
        total_step_number = int(Final_time / time_step)

        Real_part = np.real(self.wave_function)
        Imag_part = np.imag(self.wave_function)

        self.mat = np.array(self.mat)
        self.irow = np.array(self.irow)
        self.icol = np.array(self.icol)

        self.photon_mat = np.array(self.photon_mat)
        self.photon_irow = np.array(self.photon_irow)
        self.photon_icol = np.array(self.photon_icol)

        self.d1_mat = np.array(self.d1_mat)
        self.d1_irow = np.array(self.d1_irow)
        self.d1_icol = np.array(self.d1_icol)
        self.d2_mat = np.array(self.d2_mat)
        self.d2_irow = np.array(self.d2_irow)
        self.d2_icol = np.array(self.d2_icol)

        d1_energy_list = []
        d2_energy_list = []
        photon_energy_list = []

        t = 0
        Time_list = []

        wave_function_list = []
        for step in range(total_step_number):
            # evaluate result. output photon_energy, detector1_energy, detector2_energy
            if(step % output_step_number == 0 ):
                self.wave_function = np.array([np.complex(Real_part[i] , Imag_part[i]) for i in range(self.state_num)])
                # wave_function_list.append(self.wave_function)
                photon_energy1 = self.Evaluate_photon_energy()

                if(step == 0 and abs(photon_energy1 - self.photon_energy) > 0.1 ):
                    raise NameError("Error for photon energy convergence")

                detector1_energy = self.Evaluate_detector1_energy()

                detector2_energy = self.Evaluate_detector2_energy()

                d1_energy_list.append(detector1_energy)
                d2_energy_list.append(detector2_energy)
                photon_energy_list.append(photon_energy1)

                Time_list.append(t)

            # SUR algorithm

            # real_part = real_part + H * dt * imag_part
            real_part_change = self.mat * Imag_part[self.icol] * time_step
            # use numba to speed up H = H + H_change
            Real_part = wave_func_sum(Real_part, real_part_change, self.irow)

            # imag_part = imag_part - H * dt * real_part
            imag_part_change = -self.mat * Real_part[self.icol] * time_step
            # use numba to speed up
            Imag_part = wave_func_sum(Imag_part, imag_part_change, self.irow)


            t = t + time_step


        d1_energy_list = np.array(d1_energy_list)
        d2_energy_list = np.array(d2_energy_list)
        photon_energy_list = np.array(photon_energy_list)
        Time_list = np.array(Time_list)

        Total_energy = d1_energy_list  + d2_energy_list + photon_energy_list
        total_energy_len = len(Total_energy)
        # for i in range(total_energy_len):
        #     if( abs(Total_energy[i] - 1) > 0.1 ):
        #         print("simulation time step:  " + str(time_step))
        #         print('time: ' + str(Time_list[i]))
        #         print('photon energy :  ' + str(photon_energy_list[i]) + ' detector1 energy:  ' + str(d1_energy_list[i]) +"   detector2 energy:  " + str(d2_energy_list[i]) )
        #         raise NameError("SUR algorithm do not converge energy. Check code for error")


        return  photon_energy_list, d1_energy_list, d2_energy_list , Time_list

    def output_off_diagonal_coupling_mode_info(self):
        Coupling_mode_list = []
        for i in range(self.state_num, self.matnum , 2):
            irow_index = self.irow[i]
            icol_index = self.icol[i]

            coupling_mode = []
            coupling_mode.append(self.state_mode_list[irow_index])
            coupling_mode.append(self.state_mode_list[icol_index])

            Coupling_mode_list.append(coupling_mode)

        Len = len(Coupling_mode_list)
        print("Coupling number:  "  +str(Len))
        print("Coupling for state in full system: ")
        for i in range(Len):
            print(Coupling_mode_list[i])
