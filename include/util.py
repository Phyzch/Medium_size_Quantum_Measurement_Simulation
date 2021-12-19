import os
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc= comm.Get_size()

def Broadcast_data(data, num_proc):
    '''
     broadcast data to all process.
    :param data:  should have shape [iter_num_per_process, : ]
    :param num_proc:
    :return:
    '''
    data = np.array(data)
    shape = data.shape
    recv_shape = np.concatenate([[num_proc], shape])
    data_type = data.dtype

    recv_data = np.empty(recv_shape, dtype=data_type)
    comm.Gather(data, recv_data, 0)
    comm.Bcast(recv_data)

    shape_list = list(shape)
    new_shape = tuple([shape_list[0] * num_proc] + shape_list[1:])
    data = np.reshape(recv_data, new_shape)

    return data

def shuffle_data(data, num_proc , arr_random):
    '''

    :param data:
    :param num_proc:
    :param arr_random:  random number used to shuffle data. should be array with size [num_proc]
    arr_random = np.arange(num_proc)
    np.random.shuffle(arr_random)

    :return:
    '''

    data = np.array(data)
    shape = data.shape
    recv_shape = np.concatenate([[num_proc], shape])
    data_type = data.dtype

    recv_data = np.empty(recv_shape, dtype = data_type )
    comm.Gather(data, recv_data, 0 )

    recv_data_shuffle = []
    if(rank == 0):
        recv_data_shuffle = np.array( [ recv_data[i] for i in arr_random ] )

    comm.Scatter(recv_data_shuffle , data, 0 )

    return data
