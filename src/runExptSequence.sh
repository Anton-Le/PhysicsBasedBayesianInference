#!/bin/bash


#load the modules
#source "/bigdata/hplsim/scratch/lebedev/PhysicsBasedBayesianInference/src/requiredModules.lst"
module load use.own
module load compiler/python/3.8.2 mpi/mvapich/2.3.6 lib/hdf5/1.10.3_parallel lib/blis/0.6.1

#load the virtual environment
source /programs/extension/tmp/NumPyroEnv/bin/activate

cd /programs/projects/PhysicsBasedBayesianInference/src

export MPI4JAX_USE_CUDA_MPI=0
#define test parameters
highestPower=16
finalTime=210.1
stepSize=0.001
temp=1.0

numRepetitions=5

#run parameters
for rep in $(seq 1 1 $numRepetitions); do
    echo "Repetition "$rep
    for i in $(seq 1 1 $highestPower); do  
            echo "NumParticles: "$((2 ** i))
            mpiexec ./gpu_binding.sh python3 -O main.py $((2 ** i)) $finalTime $stepSize $temp
            wait
    done
    wait
done


