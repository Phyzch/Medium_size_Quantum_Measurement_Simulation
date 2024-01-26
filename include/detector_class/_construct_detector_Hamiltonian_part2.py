from include.util import *

'''
  adding coupling matrix element and reverse to diagonal form 
'''

def construct_offdiag_coupling_value(self, offdiag_coupling_element_list):
    '''
    Now we get value of off-diagonal matrix element.
    The row and column index of coupling matrix element is defined in _construct_detector_Hamiltonian_part1.py
    :return:
    '''

    # read off diagonal coupling element. deep copy to avoid encapsulate the data.
    self._offdiag_coupling_element_list = offdiag_coupling_element_list.copy()

    if type(self._offdiag_coupling_element_list) == np.ndarray:
        self._offdiag_coupling_element_list = self._offdiag_coupling_element_list.tolist()

    # construct offdiagonal coupling element
    assert len(self._offdiag_coupling_element_list) == self._offdiagonal_coupling_num, \
        'off-diagonal coupling elements input from Genetic_algorithm do not have right length'

    # dirow, dicol is already add in list in calculate_offdiag_coupling_num(self). d_H here is list.
    assert(type(self._mat) == list and type(self._offdiag_coupling_element_list) == list)
    for i in range(self._offdiagonal_coupling_num):
        self._mat[i] = self._offdiag_coupling_element_list[i + self._basis_set_state_num]


