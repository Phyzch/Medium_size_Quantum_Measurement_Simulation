import numpy as np
from include.Constructing_state_module import binary_search_mode_list
from include.Hamiltonian_class import  Hamiltonian

class detector():

    from _construct_detector_Hamiltonian_part1 import construct_detector_Hamiltonian_part1 , _construct_detector_Hamiltonian_diagonal , _construct_offdiag_dstate_coupling
    from _construct_detector_Hamiltonian_part2 import construct_offdiag_mat, reverse_dmat_diag_form

    def __init__(self , *args):
        dof, frequency, nmax, initial_state, energy_window, energy_window_for_coupling = args

        # we also have energy window cutoff for states to couple with each other
        self.energy_window_for_coupling = energy_window_for_coupling

        self.dof = dof
        self.frequency = frequency
        self.nmax = nmax

        self.initial_state = initial_state

        # give detector state energy's upper bound.
        self.energy_window = energy_window

        self.qn_diff_cutoff = 4

        self.state_energy_list = []
        self.state_mode_list = []

        self.d_H = Hamiltonian()

        self.state_num = 0
        self.offdiag_coupling_num = 0
        self.dmatnum = 0
        # off diagonal coupling element between states in molecules.
        self.offdiag_coupling_element_list = []

        self.wave_function = []

    # ------- initialize wave func ---------
    def initialize_wave_function(self):
        init_state_pos, exist = binary_search_mode_list(self.state_mode_list, self.initial_state)
        if(exist == False):
            raise NameError("Wrong . Initial state not in state space")

        self.wave_function = np.zeros(self.state_num , dtype = np.complex)
        self.wave_function[init_state_pos] = 1

        return init_state_pos

    # ---------  read and output function -----------
    def output_detector_state_coupling(self):
        '''
        output information about detector state coupling in form of [state_mode1 , state_mode2]
        :return:
        '''
        Coupling_mode_info = []
        for i in range(self.state_num , self.dmatnum):
            state_mode1 = self.state_mode_list[self.d_H.irow[i]].tolist()
            state_mode2 = self.state_mode_list[self.d_H.icol[i]].tolist()

            coupling_mode = [ state_mode1 , state_mode2 ]
            Coupling_mode_info.append(coupling_mode)

        print("detector Coupling: ")

        list_len = len(Coupling_mode_info)
        for i in range(list_len):
            print( Coupling_mode_info[i] )

    def output_offdiag_coupling_num(self):
        return self.offdiag_coupling_num

