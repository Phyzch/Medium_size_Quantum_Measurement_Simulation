from include.util import *

'''
  adding coupling matrix element and reverse to diagonal form 
'''

def construct_offdiag_mat(self, offdiag_coupling_element_list):
    '''
    Now we get off-diagonal-coupling_list_element. Continue our way of constructing Hamiltonian
    :return:
    '''

    # read off diagonal coupling element.
    self.offdiag_coupling_element_list = offdiag_coupling_element_list.copy()
    self.offdiag_coupling_element_list = self.offdiag_coupling_element_list.tolist()

    # construct offdiagonal coupling element
    if( len(self.offdiag_coupling_element_list) != self.offdiag_coupling_num ):
        raise NameError('offdiagonal coupling element input from Genetic_algorithm does not have right length')

    # dirow, dicol is already add in list in calculate_offdiag_coupling_num(self). d_H here is list.
    assert( type(self.d_H.mat) == list and type(self.offdiag_coupling_element_list) == list )
    self.d_H.mat = self.d_H.mat + self.offdiag_coupling_element_list

def reverse_dmat_diag_form(self):
    # when we use new coupling coefficient in Genetic algorithm, we have to reverse detector matrix back to diagonal form.
    self.d_H.mat = self.d_H.diag_mat.copy()
