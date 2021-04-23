import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()

def main():
    x = 1