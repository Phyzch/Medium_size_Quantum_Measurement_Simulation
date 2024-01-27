import numpy as np

from include.detector.__init__ import Detector
from include.hamiltonian_class import Hamiltonian


'''
This class serve as creating quantum system which consist of 
two medium-size-subsystem (we call it detector) and one small-size subsystem (we call it photon)
The coupling parameter between states is optimized using Genetic algorithm if we want to search best parameter to give best localization.

To use this class:
1. construct_full_system_Hamiltonian_part1
2.  run: output_offdiagonal_parameter_number(self) to tell Genetic algorithm number of off-diagonal parameter we need to feed
3. Then run construct_full_system_Hamiltonian_part2(self , offdiagonal_coupling_list) 
[This is called in Genetic algorithm fitness function]
'''


class FullSystem(Hamiltonian):
    #  ----------- import method. ---------------
    from _full_system_operation import _full_system_add_basis_set_state

    from ._construct_full_sys_hamiltonian_part1 import construct_full_system_hamiltonian_part1, \
          _compute_initial_energy, _construct_full_system_basis_set,\
         _shift_Hamiltonian,  _construct_intra_detector_coupling_submodule,  _construct_intra_detector_coupling,\
        _include_detector_detector_coupling, _construct_offdiag_detector_detector_coupling,\
        _include_photon_detector_coupling, _construct_offdiag_photon_detector_coupling

    from ._read_output_func import read_offdiag_coupling_element, \
        output_state_qn_number_list, output_off_diagonal_coupling_state_pairs_info

    from ._construct_full_sys_hamiltonian_part2 import construct_full_system_Hamiltonian_part2, _reverse_mat_diag_form, \
        _construct_full_system_offdiag_coupling

    from ._evolve_wave_func import initialize_wave_function, Evolve_dynamics, _evaluate_d_energy, \
        _evaluate_photon_energy
    # ----------------------------------------------------

    def __init__(self, detector_1_parameter, detector_2_parameter, energy_window_for_basis_set_state, photon_energy, initial_photon_wave_function, time_duration=5000, output_time_step=10):
        Hamiltonian.__init__(self)

        # Energy window for full matrix is passed through parameter directly here.
        self.time_duration = time_duration
        self.output_time_step = output_time_step

        # Instantiate the detector class for two detectors.
        # Energy_window for each detector is contained in detector_parameter.
        self.detector1 = Detector(*detector_1_parameter)
        self.detector2 = Detector(*detector_2_parameter)

        # Initialize wave function for photon
        self.initial_photon_wave_function = initial_photon_wave_function
        # Initial energy of photon + detector 1 + detector 2
        self.initial_energy = 0
        
        # For every state in full system, it composed of system state, detector 1 's state, detector 2's state.
        # state_mode_list contain information about : [ photon_mode, detector1_mode, detector2_mode ]
        self.pstate = []
        self.dstate1 = []
        self.dstate2 = []
        self.state_qn_list = []


        # System state is [00, 01, 10], thus system's state number is 3.
        self.photon_state_num = 3
        # Energy of photon
        self.init_photon_energy = photon_energy
        self.photon_state_energy = [0, photon_energy, photon_energy]
        self.photon_state_qn = [[0, 0], [1, 0], [0, 1]]

        # Energy window is used to construct state in full system relative to initial state.
        self.energy_window_for_basis_set_state = energy_window_for_basis_set_state

        self._offdiagonal_coupling_num = 0  # we only count independent off-diagonal coupling # here.

        # Off-diagonal_parameter_number is output to Genetic algorithm for optimization.
        # Off_diagonal_parameter between photon and detector
        self.pd_offdiag_param = []
        # coupling index between detector and photon
        self.pd_coupling_irow = []
        self.pd_coupling_icol = []
        self.pd_coupling_num = 0

        # Off_diagonal_parameter between detector and detector.
        self.dd_offdiag_param = []
        # coupling index between detector and detector.
        self.dd_coupling_irow = []
        self.dd_coupling_icol = []
        self.dd_coupling_num = 0

        # anharmonic coupling index in detector 1 hamiltonian and detector 2 hamiltonian.
        # use this to refer back to detector hamiltonian.
        self.d1_coupling_index = []
        self.d2_coupling_index = []

        # wave function:
        self.photon_wave_func = np.zeros(self.photon_state_num)
        self.photon_wave_func[1] = 1/np.sqrt(2)
        self.photon_wave_func[2] = 1/np.sqrt(2)
        self.wave_function = []

        # reduced density matrix for photon and detector.
        self.photon_hamiltonian_in_full_basis_set = Hamiltonian()

        self.detector1_hamiltonian_in_full_basis_set = Hamiltonian()
        self.detector2_hamiltonian_in_full_basis_set = Hamiltonian()





