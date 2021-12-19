import numpy as np


def compare_mode_quanta( mode1, mode2 ):
    dof = len(mode1)
    if dof != len(mode2):
        print("Wrong. Mode with different length can't be compared")

    for i in range(dof):
        if(mode1[i] == mode2[i]):
            continue
        elif (mode1[i] > mode2[i]):
            return 1
        elif (mode1[i] < mode2[i]):
            return -1

    return 0

# Returns index of x in arr if present, else -1
def binary_search_mode_list(arr, x ):
    if(len(arr) == 0):
        exist = False
        return 0 , exist

    Length = len(arr)
    left_flag = 0
    right_flag = Length - 1
    mid = (left_flag + right_flag) // 2


    while(right_flag > left_flag):
        mark = compare_mode_quanta(x, arr[mid])
        if(mark == 1):
            left_flag = mid + 1
            mid = (left_flag + right_flag ) // 2
        elif (mark == -1):
            right_flag = mid - 1
            mid = (left_flag + right_flag) // 2
        else:
            exist = True
            return mid , exist

    mark = compare_mode_quanta(x , arr[mid])
    if(mark == 0):
        exist  = True
        return mid , exist
    elif (mark == -1 ):
        exist = False
        return mid , exist
    else:
        exist = False
        return mid + 1 , exist




