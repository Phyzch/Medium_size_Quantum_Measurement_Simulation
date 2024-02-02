import numpy as np


def compare_quantum_number(quantum_number_1, quantum_number_2):
    dof = len(quantum_number_1)
    if dof != len(quantum_number_2):
        print("Wrong. Mode with different length can't be compared")

    for i in range(dof):
        if quantum_number_1[i] == quantum_number_2[i]:
            continue
        elif quantum_number_1[i] > quantum_number_2[i]:
            return 1
        elif quantum_number_1[i] < quantum_number_2[i]:
            return -1

    return 0

# Returns index of x in arr if present, else -1
def binary_search_qn_list(arr, element):
    '''
    binary search the quantum number in the list of quantum number. The basis set state is determined by its quantum number
    :param arr: array.
    :param element: element.
    :return:
    '''
    if len(arr) == 0:
        exist = False
        return 0 , exist

    length = len(arr)
    left_flag = 0
    right_flag = length - 1
    mid = (left_flag + right_flag) // 2


    while right_flag > left_flag:
        mark = compare_quantum_number(element, arr[mid])
        if mark == 1:
            left_flag = mid + 1
            mid = (left_flag + right_flag ) // 2
        elif  mark == -1:
            right_flag = mid - 1
            mid = (left_flag + right_flag) // 2
        else:
            exist = True
            return mid , exist

    mark = compare_quantum_number(element, arr[mid])
    if mark == 0:
        exist  = True
        return mid , exist
    elif mark == -1 :
        exist = False
        return mid , exist
    else:
        exist = False
        return mid + 1 , exist




