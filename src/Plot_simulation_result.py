import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from include.full_system_class.Full_system_class import full_system
from Fitness_function import simulate_full_system_energy_flow

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()

def Plot_simulation_result():
    # specify input paramter
    coupling_strength = 0.1

    photon_energy = 1

    dof = 3
    frequency1 = [1, 0.5, 0.25]
    frequency2 = [1, 0.5, 0.25]

    nmax1 = [1, 2, 4]
    nmax2 = [1, 2, 4]

    initial_state1 = [0, 0, 0]
    initial_state2 = [0, 0, 0]

    energy_window1 = 1
    energy_window2 = 1

    full_system_energy_window = 0

    Detector_1_parameter = dof, frequency1, nmax1, initial_state1, energy_window1
    Detector_2_parameter = dof, frequency2, nmax2, initial_state2, energy_window2

    Initial_Wavefunction = [1 / np.sqrt(2), 1 / np.sqrt(2)]

    full_system_instance = full_system(Detector_1_parameter, Detector_2_parameter, full_system_energy_window, photon_energy, Initial_Wavefunction)
    full_system_instance.construct_full_system_Hamiltonian_part1()

    off_diagonal_coupling_list = [0.06484375 , 0.01328125 , 0.04296875 , 0.03125 , 0.046875 , 0.03515625 , 0.021875 ,
                                  0.02265625 , 0.0078125 , 0.04609375 , 0.065625 , 0.040625 , 0.03125,  0.05703125,
                                  0.009375,  0.09296875 , 0.03984375,  0.021875,  0.0375,  0.0945312,  0.03828125 , 0.028125 ,
                                  0.0789062,  0.08984375 , 0.06640625 , 0.0171875 , 0.04765625,  0.065625 , 0.02734375 , 0.096875 , 0.06875 , 0.017968750000000002 , 0.0171875 , 0.02734375  ,0.00625  ,0.02109375,
                                  0.05 , 0.01484375,  0.0828125,  0.0875,  0.03359375 , 0.05625,  0.08046875 , 0.00625 ,
                                  0.06796875,  0.08671875,  0.05390625 , 0.0609375 , 0.05625,
                                  0.0546875 , 0.0265625,  0.0328125 , 0.0234375 , 0.08046875 ,
                                  0.0296875,  0.07109375 , 0.04921875,  0.0234375 , 0.03125,  0.05078125,
                                  0.08046875 , 0.04453125 , 0.01328125,  0.0640625,  0.08984375  ,0.0546875,  0.06875,
                                  0.0375, 0.02421875 , 0.0703125 , 0.0359375,  0.06328125 , 0.02734375 ,
                                  0.08828125 , 0.0625  ,0.09140625,  0.0015625  ]

    off_diagonal_coupling_list = np.array(off_diagonal_coupling_list) / 10

    photon_energy_list, d1_energy_list_change, d2_energy_list_change, Time_list  = simulate_full_system_energy_flow(full_system_instance, off_diagonal_coupling_list)

    # plot simulation result
    fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    ax1.plot(Time_list, d1_energy_list_change, label='left photon localization')
    ax1.plot(Time_list, d2_energy_list_change, label='right photon localization')
    ax1.plot(Time_list, photon_energy_list, label='photon energy')

    ax1.legend(loc='best')

    plt.show()

