import numpy as np
from Constructing_state_module import binary_search_mode_list

class detector():
    def __init__(self , dof, frequency, nmax, initial_state , energy_window):
        self.dof = dof
        self.frequency = frequency
        self.nmax = nmax
        self.initial_state = initial_state
        self.energy_window = energy_window
        self.coupling_state_distance = 3
        self.State_energy_list = []
        self.State_mode_list = []

        self.dmat = []
        self.dirow = []
        self.dicol = []

        self.dmat_diagonal = []

        self.state_num = 0
        self.offdiag_coupling_num = 0
        self.dmatnum = 0
        self.offdiag_coupling_element_list = []
        self.wave_function = []

    def construct_detector_state_sphere_cutoff(self):
        '''
               :return:
        '''
        if (len(self.initial_state) != self.dof):
            print("Wrong. initial state doesn't have right dof")

        if (len(self.frequency) != self.dof):
            print("Wrong. frequency without right dof")

        if (len(self.nmax) != self.dof):
            print("Wrong. nmax without right dof")

        mode_number = np.zeros(self.dof)
        mode_number[0] = -1

        initial_state = np.array(self.initial_state)
        initial_state_energy = np.sum(initial_state * self.frequency)

        State_energy_list = []
        State_mode_list = []
        # Define a loop to go through states available in state space:
        Bool_to_exit = False
        while (1):
            for i in range(self.dof):
                mode_number[i] = mode_number[i] + 1
                if (mode_number[i] <= self.nmax[i]):
                    break
                if (mode_number[self.dof - 1] > self.nmax[self.dof - 1]):
                    mode_number[self.dof - 1] = 0
                    Bool_to_exit = True

                mode_number[i] = 0

            # exit the while(1) cycle
            if (Bool_to_exit):
                break

            # Check if this violate energy window, if so , jump to valid state
            energy = np.sum(mode_number * self.frequency)
            if (energy > self.energy_window + initial_state_energy):
                k = 0
                while (mode_number[k] == 0):
                    mode_number[k] = self.nmax[k]
                    k = k + 1
                    if (k >= self.dof):
                        break

                if (k < self.dof):
                    mode_number[k] = self.nmax[k]

                continue

            # now put this state into list which is ordered.
            position, exist = binary_search_mode_list(State_mode_list, mode_number)
            if (exist == False):
                mode_number_copy = np.copy(mode_number)
                modenumber_copy = np.array([int(mode_number_copy[i]) for i in range(self.dof)])
                State_mode_list.insert(position, mode_number_copy)
                State_energy_list.insert(position, energy)

        self.State_energy_list = State_energy_list
        self.State_mode_list = State_mode_list
        self.state_num = len(self.State_energy_list)

    def construct_detector_Hamiltonian_diagonal(self):
        self.construct_detector_state_sphere_cutoff()
        for i in range(self.state_num):
            self.dmat.append(self.State_energy_list[i])
            self.dirow.append(i)
            self.dicol.append(i)

        self.dmat_diagonal = self.dmat.copy()

    def Reverse_dmat(self):
        self.dmat = self.dmat_diagonal.copy()


    def output_offdiag_coupling_num(self):
        return self.offdiag_coupling_num

    def read_offdiag_coupling_element(self , offdiag_coupling_element_list):
        '''

        :param offdiag_coupling_element_list: This should read from Genetic algorithm.
        :return:
        '''
        self.offdiag_coupling_element_list = offdiag_coupling_element_list.copy()

    def calculate_offdiag_coupling_num(self):
        '''
        As offdiagonal coupling should read from Genetic algorithm.
        we should calculate offdiagonal coupling number and output it
        :return:
        '''
        for i in range(self.state_num):
            for j in range( i + 1, self.state_num):
                mode_state1 = self.State_mode_list[i]
                mode_state2 = self.State_mode_list[j]
                deln = []
                deln = np.abs(np.array(mode_state1) - np.array(mode_state2) )
                mode_num_diff = np.sum ( deln  )
                if (mode_num_diff == 0):
                    raise NameError("Error. two state in detector have same mode quanta. ")

                if(mode_num_diff == 2 ):
                    mode_num_diff = 4
                    for k in range(self.dof):
                        if(deln[k] == 1):
                            deln[k] = 2
                        if(deln[k] == 2):
                            deln[k] = 4

                if(mode_num_diff == 1):
                    mode_num_diff = 3
                    for k in range(self.dof):
                        if(deln[k] == 1):
                            deln[k] = 3

                if(mode_num_diff <= self.coupling_state_distance ):
                    self.offdiag_coupling_num = self.offdiag_coupling_num + 1
                    self.dirow.append(i)
                    self.dicol.append(j)

        self.dmatnum = len(self.dirow)

    def output_detector_state_coupling(self):
        Coupling_mode_info = []
        for i in range(self.state_num , self.dmatnum):
            coupling_mode = []
            coupling_mode.append(self.State_mode_list[self.dirow[i]].tolist() )
            coupling_mode.append(self.State_mode_list[self.dicol[i]].tolist() )

            Coupling_mode_info.append(coupling_mode)

        print("detector Coupling: ")
        Len = len(Coupling_mode_info)
        for i in range(Len):
            print(Coupling_mode_info[i])

    def construct_offdiag_coupling_element(self):
        if( len(self.offdiag_coupling_element_list) != self.offdiag_coupling_num ):
            raise NameError('offdiagonal coupling element input from Genetic_algorithm does not have right length')


        # dirow, dicol is already add in list in calculate_offdiag_coupling_num(self)
        for coupling_index in range(self.offdiag_coupling_num):
            self.dmat.append(self.offdiag_coupling_element_list[coupling_index])



    def construct_detector_Hamiltonian_part1(self):
        '''
        off-diagonal coupling should read from Genetic algorithm part
        Here we only proceed to knowing off-digonal element number.
        We will need to output this number and read coupling outside class
        :return:
        '''

        # construct state
        self.construct_detector_state_sphere_cutoff()

        # construct state's Hamiltonian diagonal part
        self.construct_detector_Hamiltonian_diagonal()

        # calculate state's offdiagonal coupling
        self.calculate_offdiag_coupling_num()

    def construct_detector_Hamiltonian_part2(self):
        '''
        Now we get off-diagonal-coupling_list_element. Continue our way of constructing Hamiltonian
        :return:
        '''
        # construct offdiagonal coupling element
        self.construct_offdiag_coupling_element()

    def initialize_wave_function(self):
        position, exist = binary_search_mode_list(self.State_mode_list, self.initial_state)
        if(exist == False):
            raise NameError("Wrong . Initial state not in state space")

        self.wave_function = np.zeros(self.state_num , dtype = np.complex)
        self.wave_function[position] = 1

