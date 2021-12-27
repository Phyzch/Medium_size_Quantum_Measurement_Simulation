# Medium_size_QM_simulation_with_genetic_algorithm

### Structure of this system:

**Full_ system** : photon + 2 detector (sub quantum-system)

**photon**:

photon state has 2 dof,  basis set |LR> . |L> means it can be absorbed by left detector, |R> means it can be absorbed by right detector.

|10>  $\equiv$ |L=1, R=0> . |01> $\equiv$ |L=0, R=1> . 

Initially photon is in superposition of |10> and |01> : $|\psi \rangle = a |10\rangle + b |01\rangle$

**detector**:

detector is sub-quantum systems we build upon model of molecules. it can have  $n$ vibrational dof.  We have 2 detectors in our full_sysem : $|L\rangle$ and $|R\rangle $

### Code objective

There are 2 major objective for this code:

1. Do simulation for bunch of small quantum systems with random parameters for detector Hamiltonian sampled from distribution and decide localization statistics in statistical simulation.

    2. Use Genetic algorithm to optimize parameter to get a model with maximum localization and localization duration. (We can neglect this function now as this is just for demonstration complete localization to one side is possible by carefully choosing parameter in small quantum systems)

## System requirement

1 . 

1.1 To run this code, you have to install **mpi4py**  package in python machine.

Check : [MPI for Python &mdash; MPI for Python 3.1.3 documentation](https://mpi4py.readthedocs.io/en/stable/)

mpi4py is python implementation of MPI.

1.2. In addition, for genetic algorithm, we need to install pyeasyga package : 

[pyeasyga &mdash; pyeasyga 0.3.1 documentation](https://pyeasyga.readthedocs.io/en/latest/)

Our code define a class which is extension of class in pyeasyga.

2. For production purpose, I would advise to run code on cluster instead on local machine. 
   
   SCS lop Cluster info:  https://answers.uillinois.edu/scs/scs-clusters

        Cluster use SGE system to submit job. You have to write script to do batch simulation.  We should get together and talk about how to get cluster account for you and bash script for that.

## Usuage

```
cd  folder_path/src/
mpiexec -np 20 python3 ./main.py   
(this will run simulation with 20 process. Change 20 to num_proc you want to run )
```
