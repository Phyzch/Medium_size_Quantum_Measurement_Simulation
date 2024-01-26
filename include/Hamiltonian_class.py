import numpy as np

class Hamiltonian():

    def __init__(self):
        '''
        a base class for Hamiltonian:
        All rol, column, and matrix should be private memeber to limit its exposure to outside functions.
        '''
        # irow : row index for Hamiltonian
        self._irow = []
        # icol: column index for Hamiltonian
        self._icol = []
        # mat: matrix value for Hamiltonian.
        self._mat = []

        self._mat_array = np.zeros(1)
        self._irow_array = np.zeros(1)
        self._icol_array = np.zeros(1)

        # diag_mat : diagonal part of matrix
        self._diagonal_mat = []
        # number of states & energy of the state
        self._basis_set_state_num = 0
        self._basis_set_state_energy_list = []

        self._mat_num = 0

        # anharmonic couplings between basis set states.
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

    def show_mat_num(self):
        return self._mat_num

    def show_state_num(self):
        return self._basis_set_state_num

    def show_offdiag_matrix_num(self):
        return self._offdiagonal_coupling_num

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

    def numpy_array_for_data(self):
        '''
        convert matrix , irow, icol into numpy array.
        :return:
        '''
        self._mat_array = np.array(self._mat)
        self._irow_array = np.array(self._irow)
        self._icol_array = np.array(self._icol)
