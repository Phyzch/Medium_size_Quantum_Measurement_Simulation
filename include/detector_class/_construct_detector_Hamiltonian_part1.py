from include.util import *
from include.Constructing_state_module import binary_search_mode_list

'''
part I of constructing detector Hamiltonian. 
We construct : 1. state 2. Hamiltnian diagonal part  3. irow, icol for coupling between states in Hamiltonian. 
in part II, we will add strength of coupling between states (get from Genetic algorithm)
'''
def construct_detector_Hamiltonian_part1(self):
    '''
    off-diagonal coupling should read from Genetic algorithm part
    Here we only proceed to knowing off-digonal element number.
    We will need to output this number and read coupling outside class
    :return:
    '''

    # construct state's Hamiltonian diagonal part
    self._construct_detector_Hamiltonian_diagonal()

    # calculate state's offdiagonal coupling
    self._construct_offdiag_dstate_coupling()

def _construct_detector_Hamiltonian_diagonal(self):
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

    # Define a loop to go through states available in state space:
    Bool_to_exit = False
    while (1):
        for i in range(self.dof):
            mode_number[i] = mode_number[i] + 1
            if (mode_number[i] <= self.nmax[i]):
                break
            if (mode_number[self.dof - 1] > self.nmax[self.dof - 1]):
                Bool_to_exit = True

            mode_number[i] = 0

        # exit the while(1) cycle
        if (Bool_to_exit):
            break

        # --------- Check if this violate energy window, if so , jump to valid state -------------
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
        # -----------------------------------

        # now put this state into state_mode_list which is ordered.
        position, exist = binary_search_mode_list(self.state_mode_list, mode_number)
        if (exist == False):
            mode_number_copy = np.copy(mode_number)
            mode_number_copy = mode_number_copy.astype(int)

            self.state_mode_list.insert(position, mode_number_copy)
            self.state_energy_list.insert(position, energy)

    self.state_num = len(self.state_energy_list)

    # construct detector Hamiltonian
    for i in range(self.state_num):
        self.d_H.append(self.state_energy_list[i], i, i)
    # record diagonal part of Hamiltonian
    self.d_H.diag_mat = self.d_H.mat.copy()

def _construct_offdiag_dstate_coupling(self):
    '''
    As offdiagonal coupling strength is read from Genetic algorithm.
    we should calculate offdiagonal coupling number and output it
    :return:
    '''
    for i in range(self.state_num):
        for j in range(i + 1, self.state_num):
            mode_state1 = self.state_mode_list[i]
            mode_state2 = self.state_mode_list[j]

            # 1-norm distance between state i, j in state space
            deln = np.abs(np.array(mode_state1) - np.array(mode_state2))
            mode_num_diff = np.sum(deln)

            # ---- check diff mode_num_diff ---------
            if (mode_num_diff == 0):
                raise NameError("Error. two different state in detector have same quantum number \n")

            if (mode_num_diff == 2):
                # higher order coupling is at least cubic term. thus q.n. diff = 2 corresponds to quartic term
                mode_num_diff = 4
                for k in range(self.dof):
                    if (deln[k] == 1):
                        deln[k] = 2
                    if (deln[k] == 2):
                        deln[k] = 4

            if (mode_num_diff == 1):
                # higher order coupling is at least cubic term. thus q.n. diff = 1 corresopnds to cubic term
                mode_num_diff = 3
                for k in range(self.dof):
                    if (deln[k] == 1):
                        deln[k] = 3
            # ---------- check diff mode_num_diff -----------

            if (mode_num_diff <= self.qn_diff_cutoff):
                energy1 = self.state_energy_list[i]
                energy2 = self.state_energy_list[j]
                energy_diff = np.abs(energy1 - energy2)

                # usually we will use cutoff criteria V / \Delta E. But now V should be sampled from Genetic algorithm
                # Here instead we use energy window cutoff for criteria of coupling. We require two state connected should be near in energy
                if (energy_diff <= self.energy_window_for_coupling):
                    self.offdiag_coupling_num = self.offdiag_coupling_num + 1
                    self.d_H.irow.append(i)
                    self.d_H.icol.append(j)

    self.dmatnum = len(self.d_H.irow)