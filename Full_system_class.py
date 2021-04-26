import numpy as np
import Shared_data
from Detector_class import detector
from Constructing_state_module import binary_search_mode_list

import numba
from numba import jit

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
    len = np.shape(index)[0]
    for i in range(len):
        index_i = index[i]
        original_value[index_i] = original_value[index_i] + part_add[i]

    return original_value


class full_system():

    def __init__(self ,Detector_1_parameter, Detector_2_parameter, energy_window, photon_energy, Initial_Wave_Function):
        dof1, frequency1, nmax1, initial_state1, energy_window1 = Detector_1_parameter
        dof2, frequency2, nmax2, initial_state2, energy_window2 = Detector_2_parameter

        self.detector1 = detector(dof1,frequency1,nmax1,initial_state1,energy_window1)
        self.detector2 = detector(dof2, frequency2, nmax2, initial_state2, energy_window2)
        self.Initial_Wave_Function = Initial_Wave_Function
        self.initial_energy = 0

        self.sstate = []
        self.dstate1 = []
        self.dstate2 = []
        self.state_mode_list = []

        self.system_state_num = 3

        # energy of photon
        self.photon_energy = photon_energy

        self.system_energy = [0, photon_energy, photon_energy]
        self.system_mode = [[0,0], [1,0], [0,1]]

        self.energy_window = energy_window

        self.state_num = 0
        self.offdiagonal_parameter_number = 0
        self.matnum = 0

        self.offdiagonal_parameter_list = []
        # offdiagonal parameter between system and detector, detector & detector.
        self.offdiagonal_parameter_list_inter = []


        self.mat = []
        self.irow = []
        self.icol = []

        self.mat_diagonal_part = []

        # index for coupling in detector1 and detector2 in full system matrix.
        self.detector1_coupling_irow = []
        self.detector1_coupling_icol = []
        self.detector1_coupling_dmat_index = []

        self.detector2_coupling_irow = []
        self.detector2_coupling_icol = []
        self.detector2_coupling_dmat_index = []

        # coupling index between detector and system
        self.inter_coupling_irow = []
        self.inter_coupling_icol = []

        self.system_wave_func = np.zeros(self.system_state_num)
        self.system_wave_func[1] = 1/np.sqrt(2)
        self.system_wave_func[2] = 1/np.sqrt(2)

        self.wave_function = []

        self.mat_photon = []
        self.photon_irow = []
        self.photon_icol = []

        self.mat_detector1 = []
        self.detector1_irow = []
        self.detector1_icol = []

        self.mat_detector2 = []
        self.detector2_irow = []
        self.detector2_icol = []

        self.mat_detector1_diagonal = []
        self.mat_detector2_diagonal = []

# ----------------   first part of construcing Hamiltonian. This only have to be called once. --------------------------------
    def compute_initial_energy(self):
        self.initial_energy = 0
        detector1_energy =  np.sum(np.array(self.detector1.frequency ) * np.array(self.detector1.initial_state))
        detector2_energy = np.sum(np.array(self.detector2.frequency)  * np.array( self.detector2.initial_state) )

        self.initial_energy = self.initial_energy + self.photon_energy + detector1_energy + detector2_energy

    def construct_full_system_Hamiltonian_part1(self):
        self.compute_initial_energy()

        self.detector1.construct_detector_Hamiltonian_part1()
        self.detector2.construct_detector_Hamiltonian_part1()

        self.construct_full_system_diagonal_Hamiltonian()

        # compute offdiagonal parameter number
        self.compute_full_system_offdiagonal_parameter_number()

        # compute position of intra-detector coupling
        self.compute_position_of_intra_detector_coupling()


        # fixme: construct irow and icol. (Note for irow,icol. We add offdiagonal part between detector in compute_full_system_offdiagonal_paramter_number())
        # fixme: Then we add offdiagonal index for same detector below. Same order apply to part that reconstruct offdiagonal part of mat.
        intra_detector1_coupling_num = len(self.detector1_coupling_irow)
        for i in range(intra_detector1_coupling_num):
            self.irow.append(self.detector1_coupling_irow[i])
            self.icol.append(self.detector1_coupling_icol[i])

            #lower diangular part
            self.irow.append(self.detector1_coupling_icol[i])
            self.icol.append(self.detector1_coupling_irow[i])

        intra_detector2_coupling_num = len(self.detector2_coupling_irow)
        for i in range(intra_detector2_coupling_num):
            self.irow.append(self.detector2_coupling_irow[i])
            self.icol.append(self.detector2_coupling_icol[i])

            # lower diagonal part
            self.irow.append(self.detector2_coupling_icol[i])
            self.icol.append(self.detector2_coupling_irow[i])

        self.matnum = len(self.irow)


    def construct_full_system_diagonal_Hamiltonian(self):
        self.state_num = 0
        for i in range(self.system_state_num):
            for j in range(self.detector1.state_num):
                for k in range(self.detector2.state_num):
                    energy = self.system_energy[i] + self.detector1.State_energy_list[j] + \
                             self.detector2.State_energy_list[k]
                    if( abs(energy - self.initial_energy) <= self.energy_window ):
                        # no energy window impose
                        self.sstate.append(i)
                        self.dstate1.append(j)
                        self.dstate2.append(k)

                        state_mode =[]
                        state_mode.append(self.system_mode[i])
                        state_mode.append(self.detector1.State_mode_list[j].tolist())
                        state_mode.append(self.detector2.State_mode_list[k].tolist())
                        self.state_mode_list.append(state_mode)

                        self.mat.append( energy )
                        self.irow.append(self.state_num)
                        self.icol.append(self.state_num)
                        self.state_num = self.state_num + 1

                        self.mat_photon.append( self.system_energy[i] )
                        self.mat_detector1.append(self.detector1.State_energy_list[j])
                        self.mat_detector2.append(self.detector2.State_energy_list[k])


        self.mat_diagonal_part = self.mat.copy()

        self.photon_irow = self.irow.copy()
        self.photon_icol = self.icol.copy()

        self.detector1_irow = self.irow.copy()
        self.detector1_icol = self.icol.copy()

        self.detector2_irow = self.irow.copy()
        self.detector2_icol = self.icol.copy()

        self.mat_detector1_diagonal = self.mat_detector1.copy()
        self.mat_detector2_diagonal = self.mat_detector2.copy()

    def print_state_mode(self):
        print(self.state_mode_list)

    def compute_position_of_intra_detector_coupling(self):
        for i in range(self.state_num):
            for j in range(i + 1, self.state_num):
                ss = self.sstate[i] - self.sstate[j]

                di1 = self.dstate1[i]
                di2 = self.dstate2[i]
                dj1 = self.dstate1[j]
                dj2 = self.dstate2[j]

                if (self.sstate[i] + self.sstate[j] == 3):
                    ss = -3

                # coupling in detector2
                if (ss == 0 and di1 == dj1 and di2 != dj2):
                    for k in range(self.detector2.state_num, self.detector2.dmatnum):
                        if (self.detector2.dirow[k] == di2 and self.detector2.dicol[k] == dj2):
                            self.detector2_coupling_irow.append(i)
                            self.detector2_coupling_icol.append(j)
                            self.detector2_coupling_dmat_index.append(k)

                            self.detector2_irow.append(i)
                            self.detector2_icol.append(j)

                            # also record lower triangular part
                            self.detector2_icol.append(i)
                            self.detector2_irow.append(j)

                            break

                # coupling in detector 1
                elif (ss == 0 and di1 != dj1 and di2 == dj2):
                    for k in range(self.detector1.state_num, self.detector1.dmatnum):
                        if (self.detector1.dirow[k] == di1 and self.detector1.dicol[k] == dj1):
                            self.detector1_coupling_irow.append(i)
                            self.detector1_coupling_icol.append(j)
                            self.detector1_coupling_dmat_index.append(k)

                            self.detector1_irow.append(i)
                            self.detector1_icol.append(j)

                            # also record lower triangular part
                            self.detector1_icol.append(i)
                            self.detector1_irow.append(j)


                            break

    def compute_full_system_offdiagonal_parameter_number(self):
        self.offdiagonal_parameter_number = self.offdiagonal_parameter_number +  self.detector1.offdiag_coupling_num
        self.offdiagonal_parameter_number = self.offdiagonal_parameter_number + self.detector2.offdiag_coupling_num

        coupling_num_photon_detector = 0
        coupling_num_between_detector = 0

        # count coupling between system and detector
        for i in range(self.state_num):
            for j in range( i+1 , self.state_num):
                ss = self.sstate[i] - self.sstate[j]

                di1 = self.dstate1[i]
                di2 = self.dstate2[i]
                dj1 = self.dstate1[j]
                dj2 = self.dstate2[j]

                if(self.sstate[i] + self.sstate[j] == 3):
                    ss= -3

                # coupling for photon with detector1
                if(ss == -1 and di1 != dj1 and di2 == dj2):
                    if(self.detector1.State_mode_list[di1][0] - self.detector1.State_mode_list[dj1][0] == 1 ):
                        Same = True
                        for k in range(1, self.detector1.dof):
                            if(self.detector1.State_mode_list[di1][k] != self.detector1.State_mode_list[dj1][k]):
                                Same = False
                                break

                        if(Same):
                            self.offdiagonal_parameter_number = self.offdiagonal_parameter_number + 1
                            self.irow.append(i)
                            self.icol.append(j)
                            # lower triangular part.
                            self.irow.append(j)
                            self.icol.append(i)

                            self.inter_coupling_irow.append(i)
                            self.inter_coupling_icol.append(j)

                            coupling_num_photon_detector = coupling_num_photon_detector + 1

                # coupling for photon with detector2
                if(ss == -2 and di1 == dj1 and di2 != dj2):
                    if(self.detector2.State_mode_list[di2][0] - self.detector2.State_mode_list[dj2][0] == 1):
                        Same = True
                        for k in range(1, self.detector2.dof):
                            if(self.detector2.State_mode_list[di2][k] != self.detector2.State_mode_list[dj2][k]):
                                Same = False
                                break

                        if(Same):
                            self.offdiagonal_parameter_number = self.offdiagonal_parameter_number + 1
                            self.inter_coupling_irow.append(i)
                            self.inter_coupling_icol.append(j)

                            self.irow.append(i)
                            self.icol.append(j)
                            # lower triangular part.
                            self.irow.append(j)
                            self.icol.append(i)

                            coupling_num_photon_detector = coupling_num_photon_detector + 1

                # coupling between detector1 and detector2
                if(ss ==0 and di1!= dj1 and di2 != dj2):
                    for k in range(1,self.detector2.dof):
                        deldv2 = self.detector2.State_mode_list[di2][k] - self.detector2.State_mode_list[dj2][k]
                        if( abs(deldv2) == 1 ):
                            zero = 0
                            for l in range(k):
                                zero = zero + abs(self.detector2.State_mode_list[di2][l] - self.detector2.State_mode_list[dj2][l])

                            for l in range(k+1, self.detector2.dof):
                                zero = zero +  abs(self.detector2.State_mode_list[di2][l] - self.detector2.State_mode_list[dj2][l])

                            if (zero != 0):
                                break

                            for k1 in range(1,self.detector1.dof):
                                deldv1 = self.detector1.State_mode_list[di1][k1] - self.detector1.State_mode_list[dj1][k1]
                                if(abs(deldv1) == 1):
                                    zero = 0
                                    for l in range(k1):
                                        zero = zero + abs(self.detector1.State_mode_list[di1][l] - self.detector1.State_mode_list[dj1][l])

                                    for l in range(k1 + 1, self.detector1.dof):
                                        zero = zero + abs(self.detector1.State_mode_list[di1][l]  -self.detector1.State_mode_list[dj1][l] )

                                    if(zero!= 0):
                                        break

                                    self.offdiagonal_parameter_number =self.offdiagonal_parameter_number + 1
                                    self.inter_coupling_irow.append(i)
                                    self.inter_coupling_icol.append(j)

                                    self.irow.append(i)
                                    self.icol.append(j)
                                    # lower triangular part.
                                    self.irow.append(j)
                                    self.icol.append(i)

                                    coupling_num_between_detector = coupling_num_between_detector + 1


#  --------------------------------------- first part of construcing Hamiltonian. This only have to be called once. End ---------------------

 # -------------------------- Read and output offdiagonal parameter number . Also reverse matrix begin ---------------------
    def output_offdiagonal_parameter_number(self):
        # we need to output offdiagonal parameter number to tell Genetic algorithm how many parameters we need to sample
        return self.offdiagonal_parameter_number

    def read_offdiag_coupling_element(self,offdiagonal_coupling_list):
        self.offdiagonal_parameter_list = offdiagonal_coupling_list.copy()

        begin_index = 0
        end_index = self.detector1.offdiag_coupling_num
        off_diagonal_parameter_for_detector1 = offdiagonal_coupling_list [ begin_index: end_index].copy()

        begin_index = begin_index + self.detector1.offdiag_coupling_num
        end_index = end_index + self.detector2.offdiag_coupling_num
        off_diagonal_parameter_for_detector2 = offdiagonal_coupling_list [ begin_index : end_index ].copy()

        begin_index = begin_index + self.detector2.offdiag_coupling_num
        end_index = self.offdiagonal_parameter_number
        self.offdiagonal_parameter_list_inter = offdiagonal_coupling_list[begin_index : end_index].copy()

        self.detector1.read_offdiag_coupling_element(off_diagonal_parameter_for_detector1)
        self.detector2.read_offdiag_coupling_element(off_diagonal_parameter_for_detector2)

    def Reverse_mat(self):
        # For each generation, we only have to update off-diagonal part . We do not have to compute off-diagonal coupling number and reconstruct diagonal part
        self.detector1.Reverse_dmat()
        self.detector2.Reverse_dmat()

        self.mat = self.mat_diagonal_part.copy()

        self.mat_detector1 = self.mat_detector1_diagonal.copy()
        self.mat_detector2 = self.mat_detector2_diagonal.copy()


    # -------------------------- Read and output offdiagonal parameter number . Also reverse matrix  End---------------------

    def construct_full_system_offdiag_coupling(self):

        inter_detector_coupling_num = len(self.inter_coupling_irow)
        if(inter_detector_coupling_num != len(self.offdiagonal_parameter_list_inter)):
            raise NameError("inter detector coupling number does not equal to parameter number read from Genetic algorithm")

        # coupling between detector and system. and detector between detector
        for i in range(inter_detector_coupling_num):
            self.mat.append(self.offdiagonal_parameter_list_inter[i])

            self.mat.append(self.offdiagonal_parameter_list_inter[i])

        # coupling in detector 1
        intra_detector1_coupling_num = len(self.detector1_coupling_irow)
        for i in range(intra_detector1_coupling_num):
            k = self.detector1_coupling_dmat_index[i]

            self.mat.append(self.detector1.dmat[ k ])

            # we also record lower trangular part
            self.mat.append(self.detector1.dmat[ k ])


            # We construct Hamiltonian for detector1
            self.mat_detector1.append(self.detector1.dmat[ k ])
            # also lower triangular part
            self.mat_detector1.append(self.detector1.dmat[ k ])


        # coupling in detector2
        intra_detector2_coupling_num = len(self.detector2_coupling_irow)
        for i in range(intra_detector2_coupling_num):
            k = self.detector2_coupling_dmat_index[i]

            self.mat.append(self.detector2.dmat[ k ])

            self.mat.append(self.detector2.dmat[ k ])

            # We construct Hamiltonian for detector2
            self.mat_detector2.append(self.detector2.dmat[ k ])
            # also lower triangular part
            self.mat_detector2.append(self.detector2.dmat[ k ])



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


        # each detector construct their hamiltonian
        self.detector1.construct_detector_Hamiltonian_part2()
        self.detector2.construct_detector_Hamiltonian_part2()

        # full system construct Hamiltonian using detector's Hamiltonian.
        self.construct_full_system_offdiag_coupling()

        # initialize wave function.
        self.initialize_wave_function()

        # shift Hamiltonian
        self.Shift_Hamiltonian()


    def initialize_wave_function(self):
        self.detector1.initialize_wave_function()
        self.detector2.initialize_wave_function()

        position1, exist1 = binary_search_mode_list(self.detector1.State_mode_list, self.detector1.initial_state)
        position2, exist2 = binary_search_mode_list(self.detector2.State_mode_list, self.detector2.initial_state)

        self.wave_function = np.zeros(self.state_num, dtype = np.complex)

        for i in range(self.state_num):
            if self.sstate[i] == 1  :
                if self.dstate1[i] == position1 and self.dstate2[i] == position2 :
                    self.wave_function[i] = self.Initial_Wave_Function[0]

            if self.sstate[i] == 2 :
                if self.dstate1[i] == position1 and self.dstate2[i] == position2 :
                    self.wave_function[i] = self. Initial_Wave_Function[1]

    def Shift_Hamiltonian(self):
        for i in range(self.state_num):
            self.mat[i] = self.mat[i] - self.initial_energy

    def Evaluate_photon_energy(self):
        # use self.mat_photon and self.photon_irow. self.photon_icol
        H_phi = self.mat_photon * self.wave_function[self.photon_icol]

        H_phi_wave_function = np.zeros(self.state_num,dtype=np.complex)
        H_phi_wave_function = wave_func_sum(H_phi_wave_function,H_phi, self.photon_irow)

        photon_energy = np.sum (np.real(np.conjugate(self.wave_function) * H_phi_wave_function) )

        return photon_energy

    def Evaluate_detector1_energy(self):
        H_phi = self.mat_detector1 * self.wave_function[self.detector1_icol]

        H_phi_wave_function = np.zeros(self.state_num,dtype=np.complex)
        H_phi_wave_function = wave_func_sum(H_phi_wave_function, H_phi, self.detector1_irow)

        detector1_energy = np.sum( np.real( np.conjugate(self.wave_function) * H_phi_wave_function ))

        return detector1_energy

    def Evaluate_detector2_energy(self):
        H_phi = self.mat_detector2 * self.wave_function[self.detector2_icol]

        H_phi_wave_function = np.zeros(self.state_num,dtype=np.complex)
        H_phi_wave_function = wave_func_sum(H_phi_wave_function, H_phi, self.detector2_irow)

        detector2_energy = np.sum(np.real(np.conjugate(self.wave_function) * H_phi_wave_function))

        return detector2_energy

    def Evolve_dynamics(self):
        Final_time = Shared_data.Time_duration
        output_time_step = Shared_data.output_time_step

        # define time step to do simulation
        Max_element = np.max( np.abs(self.mat) )
        time_step = 0.1 / Max_element

        # output step number and total_step_number
        output_step_number = max( int(output_time_step / time_step) , 1)
        total_step_number = int(Final_time / time_step)

        Real_part = np.real(self.wave_function)
        Imag_part = np.imag(self.wave_function)
        # SUR algorithm
        # Vectorize

        self.mat = np.array(self.mat)
        self.irow = np.array(self.irow)
        self.icol = np.array(self.icol)

        self.mat_photon = np.array(self.mat_photon)
        self.photon_irow = np.array(self.photon_irow)
        self.photon_icol = np.array(self.photon_icol)

        self.mat_detector1 = np.array(self.mat_detector1)
        self.detector1_irow = np.array(self.detector1_irow)
        self.detector1_icol = np.array(self.detector1_icol)
        self.detector2_irow = np.array(self.detector2_irow)
        self.detector2_icol = np.array(self.detector2_icol)

        d1_energy_list = []
        d2_energy_list = []
        photon_energy_list = []

        t = 0
        Time_list = []

        for step in range(total_step_number):

            # SUR algorithm

            # real_part = real_part + H * dt * imag_part
            # imag_part = imag_part - H * dt * real_part
            real_part_change = self.mat * Imag_part[self.icol] * time_step
            # use numba to speed up
            Real_part = wave_func_sum(Real_part, real_part_change, self.irow)

            imag_part_change = -self.mat * Real_part[self.icol] * time_step
            # use numba to speed up
            Imag_part = wave_func_sum(Imag_part, imag_part_change, self.irow)

            # evaluate result. output photon_energy, detector1_energy, detector2_energy
            if(step % output_step_number == 0 ):
                self.wave_function = np.array([np.complex(Real_part[i] , Imag_part[i]) for i in range(self.state_num)])

                photon_energy = self.Evaluate_photon_energy()

                if(step == 0 and abs(photon_energy - 1) > 0.1 ):
                    print("Error")

                detector1_energy = self.Evaluate_detector1_energy()

                detector2_energy = self.Evaluate_detector2_energy()

                d1_energy_list.append(detector1_energy)
                d2_energy_list.append(detector2_energy)
                photon_energy_list.append(photon_energy)

                Time_list.append(t)

            t = t + time_step


        d1_energy_list = np.array(d1_energy_list)
        d2_energy_list = np.array(d2_energy_list)
        photon_energy_list = np.array(photon_energy_list)
        Time_list = np.array(Time_list)

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
