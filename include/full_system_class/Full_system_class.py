import numpy as np
from include.detector_class.Detector_class import detector

'''
This class serve as creating quantum system which consist of 
two medium-size-subsystem (we call it detector) and one small-size subsystem (we call it photon)
The coupling parameter between states is optimized using Genetic algorithm if we want to search best parameter to give best localization.

To use this class:
1. construct_full_system_Hamiltonian_part1
2.  run: output_offdiagonal_parameter_number(self) to tell Genetic algorithm number of off-diagonal parameter we need to feed
3. Then run construct_full_system_Hamiltonian_part2(self , offdiagonal_coupling_list) [This is called in Genetic algorithm fitness function]
'''

class Hamiltonian():
    # a primitive class for Hamiltonian:
    # irow : row index for Hamiltonian
    irow = []
    # icol: column index for Hamiltonian
    icol = []
    # mat: matrix value for Hamiltonian.
    mat = []

    # diag_mat : diagonal part of matrix
    diag_mat = []

    def append( self, mat_element, irow_index, icol_index):
        self.mat.append(mat_element)
        self.irow.append(irow_index)
        self.icol.append(icol_index)

    def matnum(self):
        return len(self.mat)

    def statenum(self):
        return len(self.diag_mat)

    def data_to_numpy(self):
        self.mat = np.array(self.mat)
        self.irow = np.array(self.irow)
        self.icol = np.array(self.icol)


class full_system():
    Time_duration = 50000
    output_time_step = 100

    from ._construct_full_sys_hamiltonian_part1 import construct_full_system_Hamiltonian_part1
    from ._read_output_func import read_offdiag_coupling_element, output_offdiagonal_parameter_number, output_state_mode, output_off_diagonal_coupling_mode_info
    from ._construct_full_sys_hamiltonian_part2 import construct_full_system_Hamiltonian_part2
    from ._evolve_wave_func import initialize_wave_function , Evolve_dynamics, __evaluate_d_energy , __evaluate_photon_energy

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

        # offdiagonal_parameter_number is output to Genetic algorithm for optimization.
        # off_diagonal_parameter in list is in order: intra-detector coupling in detector 1, intra-detector couping in detector 2. coupling between detector and photon , coupling between detector and deteoctr.
        self.offdiag_param_num = 0
        # off diagonal parameter : 1. parameter for intra-detector coupling  2. coupling between detectors.
        self.offdiag_param_list = []
        # offdiagonal parameter : 1. between system and detector and 2.between detector and detector.
        self.pd_dd_offdiag_param = []

        # full_H : full_Hamiltonian for (photon + detector)
        self.full_H = Hamiltonian()

        # off diagonal coupling element in detector1 and detector2 , their corresponding index in full system matrix :
        self.d1_coupling_H = Hamiltonian()
        self.d1_coupling_dmat_index = []
        
        self.d2_coupling_H = Hamiltonian()
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
        self.photon_H = Hamiltonian()

        # Hamiltonian solely for detector1 in full matrix
        self.d1_H = Hamiltonian()

        # Hamiltonian solely for detectro2 in full matrix
        self.d2_H = Hamiltonian()



