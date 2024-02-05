import numpy as np

class Hamiltonian:
    def __init__(self):
        '''
        a base class for Hamiltonian:
        All rol, column, and matrix should be private member to limit its exposure to outside functions.
        '''
        # we choose coordinate list (COO) form to store sparse matrix.
        # irow : row index for Hamiltonian
        self._irow = []
        # icol: column index for Hamiltonian
        self._icol = []
        # mat: matrix value for Hamiltonian.
        self._mat = []

        # diagonal_mat : diagonal part of matrix
        self._diagonal_mat = []

        # number of states & energy of basis set states
        self._basis_set_state_num = 0
        self._basis_set_state_energy_list = []

        # total number of sparse matrix.
        self._mat_num = 0

        # off-diagonal couplings between basis set states.
        self._offdiagonal_coupling_num = 0

        # anharmonic coupling elements between states in molecules.
        self._offdiag_coupling_element_list = []

    def append_matrix_element( self, mat_element, irow_index, icol_index):
        '''
        insert one matrix element into the sparse matrix of Hamiltonian.
        :param mat_element:
        :param irow_index:
        :param icol_index:
        :return:
        '''
        assert(type(self._mat) == list and type(self._irow) == list and type(self._icol) == list)
        self._mat.append(mat_element)
        self._irow.append(irow_index)
        self._icol.append(icol_index)

    def _add_basis_set_energy_to_hamiltonian_diagonal_part(self):
        '''
        construct diagonal part of the hamiltonian.
        :return:
        '''
        # construct detector Hamiltonian diagonal part.
        for i in range(self._basis_set_state_num):
            self.append_matrix_element(self._basis_set_state_energy_list[i], i, i)

        # record diagonal part of Hamiltonian
        self._diagonal_mat = self._mat.copy()


    def show_mat_num(self):
        '''
        return total number of hamiltonian matrix.
        :return:
        '''
        return self._mat_num

    def show_state_num(self):
        '''
        return number of basis set states.
        :return:
        '''
        return self._basis_set_state_num

    def show_offdiag_matrix_num(self):
        '''
        return number of off-diagonal matrix elements.
        :return:
        '''
        return self._offdiagonal_coupling_num

    def get_basis_set_state_energy(self, i):
        '''
        return energy of the basis set state.
        :param i: index for the element.
        :return:
        '''
        return self._basis_set_state_energy_list[i]


    def get_irow(self, i):
        '''
        get row index of sparse Hamiltonian matrix with element index i.
        :param i: element index (including diagonal and off-diagonal component of Hamiltonian)
        :return:
        '''
        return self._irow[i]

    def get_icol(self, i):
        '''
        get column index of sparse Hamiltonian matrix with element index i.
        :param i: element index (including diagonal and off-diagonal component of Hamiltonian)
        :return:
        '''
        return self._icol[i]

    def get_matrix_value(self, i):
        '''
        get matrix value of sparse Hamiltonian matrix with element index i.
        :param i: element_index (including diagonal and off-diagonal component of Hamiltonian)
        :return:
        '''
        return self._mat[i]

    def get_mat_array(self):
        '''
        show numpy array of hamiltonian matrix
        :return:
        '''
        return np.copy(self._mat)

    def get_irow_array(self):
        '''
        show numpy array of the row index
        :return:
        '''
        return np.copy(self._irow)

    def get_icol_array(self):
        '''
        show numpy array of the column index
        :return:
        '''
        return np.copy(self._icol)

    def replace_mat_value(self, i, value):
        '''
        replace matrix value of sparse Hamiltonian matrix with element index i.
        :param i: index to replace
        :param value: new value of matrix element.
        :return:
        '''
        self._mat[i] = value

    def numpy_array_for_data(self):
        '''
        convert matrix , irow, icol into numpy array.
        :return:
        '''
        mat_array = np.array(self._mat)
        irow_array = np.array(self._irow)
        icol_array = np.array(self._icol)

        return mat_array, irow_array, icol_array