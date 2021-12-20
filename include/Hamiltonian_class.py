import numpy as np

class Hamiltonian():
    # a primitive class for Hamiltonian:
    # irow : row index for Hamiltonian
    irow = []
    # icol: column index for Hamiltonian
    icol = []
    # mat: matrix value for Hamiltonian.
    mat = []

    mat_array = np.zeros(1)
    irow_array = np.zeros(1)
    icol_array = np.zeros(1)

    # diag_mat : diagonal part of matrix
    diag_mat = []

    def append( self, mat_element, irow_index, icol_index):
        assert( type(self.mat) == list and type(self.irow) == list and type(self.icol) == list )
        self.mat.append(mat_element)
        self.irow.append(irow_index)
        self.icol.append(icol_index)

    def matnum(self):
        return len(self.mat)

    def statenum(self):
        return len(self.diag_mat)

    def data_to_numpy(self):
        self.mat_array = np.array(self.mat)
        self.irow_array = np.array(self.irow)
        self.icol_array = np.array(self.icol)
