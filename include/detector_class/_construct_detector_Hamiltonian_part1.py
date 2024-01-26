from include.util import *
from include.Constructing_state_module import binary_search_qn_list


'''
part I of constructing detector Hamiltonian. 
We construct : 1. state 2. Hamiltonian diagonal part  3. irow, icol for coupling between states in Hamiltonian. 
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
    self._construct_detector_Hamiltonian_diagonal_part()

    # calculate state's anharmonic coupling
    self._construct_offdiag_detector_state_coupling()

def _construct_detector_Hamiltonian_diagonal_part(self):
    assert len(self._initial_state_qn) == self._dof, print("Wrong. initial state doesn't have right dof")

    assert len(self._frequency) == self._dof, print("Wrong. frequency array does not have right dof")

    assert len(self._nmax) != self._dof, print("Wrong. nmax array does not right dof")

    quantum_number = np.zeros(self._dof)
    quantum_number[0] = -1

    initial_state_qn = np.array(self._initial_state_qn)
    initial_state_energy = np.sum(initial_state_qn * self._frequency)

    # Define a loop to go through states available in state space:
    exit_loop_bool = False
    while 1:
        for i in range(self._dof):
            quantum_number[i] = quantum_number[i] + 1
            if quantum_number[i] <= self._nmax[i]:
                break
            if quantum_number[self._dof - 1] > self._nmax[self._dof - 1]:
                exit_loop_bool = True

            quantum_number[i] = 0

        # exit the while(1) cycle
        if exit_loop_bool:
            break

        # --------- Check if this state is outside energy window, if so , jump to valid state -------------
        energy = np.sum(quantum_number * self._frequency)
        if  energy > self._energy_window + initial_state_energy:
            k = 0
            # jump to next state whose energy is smaller than initial_state_energy + energy_window
            while quantum_number[k] == 0:
                quantum_number[k] = self._nmax[k]
                k = k + 1
                if k >= self._dof:
                    break

            if k < self._dof:
                quantum_number[k] = self._nmax[k]

            continue
        # -----------------------------------

        # now put this state into state_mode_list which is ordered.
        position, exist = binary_search_qn_list(self._basis_set_state_qn_list, quantum_number)
        if not exist:
            self.detector_add_basis_set_state(quantum_number, energy, position)

    self._basis_set_state_num = len(self._basis_set_state_energy_list)

    # construct diagonal part of the hamiltonian.
    self._detector_add_hamiltonian_diagonal_part()

def _construct_offdiag_detector_state_coupling(self):
    '''
    As off-diagonal coupling strength is read from Genetic algorithm.
    we should calculate off-diagonal coupling number and output it
    :return:
    '''

    offdiag_hamiltonian_init_value = -1.0

    for i in range(self._basis_set_state_num):
        for j in range(i + 1, self._basis_set_state_num):
            state1_qn = self._basis_set_state_qn_list[i]
            state2_qn = self._basis_set_state_qn_list[j]

            # 1-norm distance between state i, j in state space
            deln = np.abs(np.array(state1_qn) - np.array(state2_qn))
            qn_num_diff = np.sum(deln)

            # ---- check diff mode_num_diff ---------
            assert qn_num_diff != 0 , "Error. two different state in detector have same quantum number \n"

            if qn_num_diff == 2:
                # higher order coupling is at least cubic term. thus q.n. diff = 2 corresponds to quartic term
                qn_num_diff = 4
                for k in range(self._dof):
                    if deln[k] == 1:
                        deln[k] = 2
                    if deln[k] == 2:
                        deln[k] = 4

            if qn_num_diff == 1:
                # higher order coupling is at least cubic term. thus q.n. diff = 1 corresopnds to cubic term
                qn_num_diff = 3
                for k in range(self._dof):
                    if deln[k] == 1:
                        deln[k] = 3
            # ---------- check diff qn_num_diff -----------
            if qn_num_diff <= self._qn_diff_cutoff:
                energy1 = self._basis_set_state_energy_list[i]
                energy2 = self._basis_set_state_energy_list[j]
                energy_diff = np.abs(energy1 - energy2)

                # usually we will use cutoff criteria V / \Delta E. But now V should be sampled from Genetic algorithm
                # Here instead we use energy window cutoff for criteria of anharmonic coupling.
                # We require two state connected should be near in energy
                if energy_diff <= self._energy_window_for_coupling:
                    # here we append off-diagonal hamiltonian element
                    # (independent off-diagonal element, if we include (i,j), we skip (j,i)) to the hamiltonian matrix.
                    # but the matrix element value is un-defined.
                    self._offdiagonal_coupling_num = self._offdiagonal_coupling_num + 1
                    self.append_matrix_element(offdiag_hamiltonian_init_value, i, j)


    self._mat_num = self._offdiagonal_coupling_num + self._basis_set_state_num