from Feed_full_system_to_Genetic_algorithm import Implement_genetic_algorithm
from Born_rule import Analyze_Born_rule
from Plot_simulation_result import Plot_simulation_result
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_proc = comm.Get_size()

def main():
    file_path = "/home/phyzch/PycharmProjects/Medium_size_Quantum_Measurement_simulation/result/Genetic_algorithm/3_mode/"
    Implement_genetic_algorithm(file_path)

    # Plot_simulation_result()

    Analyze_Born_rule_file_path = "/home/phyzch/PycharmProjects/Medium_size_Quantum_Measurement_simulation/result/Born_rule/try/"
    Analyze_Born_rule(Analyze_Born_rule_file_path)

main()