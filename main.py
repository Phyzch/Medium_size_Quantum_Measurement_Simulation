from Feed_full_system_to_Genetic_algorithm import Implement_genetic_algorithm
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()

def main():
    file_path = "/home/phyzch/PycharmProjects/Medium_size_Quantum_Measurement_simulation/result/try/"
    Implement_genetic_algorithm(file_path)

main()