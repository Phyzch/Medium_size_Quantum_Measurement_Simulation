import os
import numpy as np
import copy
import random
from timeit import default_timer as timer

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc= comm.Get_size()

def broadcast_data(data, num_proc):
    '''
     broadcast data to all process.
    :param data:  should have shape [iter_num_per_process, : ]
    :param num_proc: number of process for MPI calculation.
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

def shuffle_data(data, num_proc, random_arr):
    '''
    shuffle the data in each processes to other processes according to order (random_arr)
    :param data: data to shuffle
    :param num_proc: number of process in parallel computing
    :param random_arr:  random number used to shuffle data. should be array with size [num_proc]
    random_arr = np.arange(num_proc)
    np.random.shuffle(random_arr)
    :return:
    '''
    data = np.array(data)
    shape = data.shape
    recv_shape = np.concatenate([[num_proc], shape])
    data_type = data.dtype

    recv_data = np.empty(recv_shape, dtype = data_type )
    comm.Gather(data, recv_data, 0 )

    recv_data_shuffle = []
    if rank == 0:
        recv_data_shuffle = np.array([recv_data[i] for i in random_arr])

    comm.Scatter(recv_data_shuffle , data, 0 )

    return data
