#!/usr/bin/env python3
# This file contains integrated tests of
# operations required to parallelise SMC resampling
# It is a deterministic sample implementation
# of resampling

import numpy as np
import random
import jax
import mpi4jax
import numpyro
import mpi4py
from mpi4py import MPI, rc


rc.threaded=True
rc.thread_level="funneled"
comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()


def generateWeights(processRank:int):
    """
    This function takes a proces rank {0,1,2} and returns
    a predetermined set of locally normalized weights.
    """
    assert (0 <= processRank and processRank <= 2), "Wrong process rank!"
    w = np.zeros( 4 )
    if processRank == 0:
        w = np.array([1/33,1/33,30/33, 1/33])
    elif processRank == 1:
        w = np.array([0,0, 1/2, 1/2])
    else:
        w = np.array([1/4,1/8,3/8,1/4])
    return w
def generateVectors(processRank:int):
    """
    Generate a set of 4 3-vectors representing
    the positions of the particles with the weights
    provided via the `generateWeights` function.
    """
    assert (0 <= processRank and processRank <= 2), "Wrong process rank!"
    #positions as columns of a matrix
    e = np.ones( 3 )
    scale = np.ones( 4 )
    if processRank == 0:
        scale = np.array([4,3,2,1])
    elif processRank == 1:
        scale = np.array([3,1,2,4])
    else:
        scale = np.array([1,4,2,3])

    pos = np.outer(e, scale)
    return pos
def generateReferenceCDF():
    """
    This function returns an array containing a CDF
    representing the distribution of weights provided with
    `generateWeights`.
    """
    F = np.array([1/99,2/99,32/99,1/3,1/3,1/3,1/2,2/3,3/4,19/24,11/12,1] )
    return F

def test_gather():
    """
    This function is used to gather 
    """
    assert size == 3, "The MPI tests require 3 processes!"
    #generate a uniform distribution of weights, 10 each
    #modified by the rank
    weights = generateWeights( rank )

    globalWeights, _ = mpi4jax.gather(weights, root=0)
    for i in range(size):
        if rank == i:
            print("Process ", i, "w = ", weights)

    if rank == 0:
             globalWeights = globalWeights.flatten()
             print("Global weights: ", globalWeights)
             cdf = np.cumsum(globalWeights / np.sum(globalWeights) )
             print("CDF: ", cdf)
             referenceCDF = generateReferenceCDF()
             print("Correct CDF? ", np.allclose(cdf, referenceCDF) )
    return


def test_CoreResampling():
    """
    This is a bare-bones implementation of an MPI-parallelised
    resampling usable for Sequential Monte Carlo.
    """
    # Step 1 - generate local data
    weights_local = generateWeights( rank )
    q_local = generateVectors( rank )
    Nparticles_global = 12
    Nparticles_local = 4
    Nproc = 3
    # Step 1.5 - sort particles and weights in descending order
    sortedIndices = np.argsort( weights_local )
    reversedSortedIndices = sortedIndices[ ::-1 ]
    weights_local = weights_local[ reversedSortedIndices ]
    q_local = q_local[ reversedSortedIndices, : ]

    # Step 2 - gather the local weights onto all processes
    weights_global, _ = mpi4jax.allgather( weights_local )
    # Step 3 - compute the CDF on each process
    cdf = np.cumsum( weights_global / np.sum(weights_global) )
    # Step 4 - define an array of 'random numbers'
    #sufficient to resample all local particles.
    resamplingRNs = None
    if rank == 0:
        resamplingRNs = np.random.uniform(0,1, size=Nparticles_local)
    # Step 5 - broadcast said array.
    resamplingRNs, _ = mpi4jax.bcast(resamplingRNs, 0)
    # Step 6 - every process performs CDF inversion
    newSample_globalIndices = np.zeros(Nparticles_local, dtype=int)
    for idx in range(Nparticles_local):
        newSample_globalIndices[idx] = np.argmax( resamplingRNs[idx] <= cdf )
    # Step 7 - determine the process indices of the resampled particles
    processIndices = newSample_globalIndices // Nparticles_local
    # Step 8 - count the number of particles from each process
    particlesFromProcess = np.bincount( processIndices, minlength=Nproc)
    # Step 9 - Gather particles unto all - into a flat buffer
    # TODO: Check functionality separately!
    sendbuf = q_local[ :particlesFromProcess[rank], :]
    sendbuf = sendbuf.flatten()
    recvbuf = np.empty( Nparticles_local * q_local.shape[1], dtype=q_local.dtype )
    recvDisplacements = np.zeros( Nparticles_local, dtype=int)
    for pId in range(1, Nparticles_local):
            recvDisplacements[pId] = particlesFromProcess[pId - 1]
    recvDisplacements = np.cumsum(recvDisplacements)
    comm.Allgatherv( sendbuf=(sendbuf, particlesFromProcess * Nparticles_local, MPI.DOUBLE),\
                     recvbuf=(recvbuf, particlesFromProcess * Nparticles_local, recvDisplacements, MPI.DOUBLE))
    return



def test_gatherv():
    """
    Test of the MPI Gatherv function for the collection of
    a subset of positions and momenta from the workers.
    Since MPI4JAX does not provide this function we
    need to fall back to MPI4Py.
    """
    #generate the data
    w_local = generateWeights( rank )
    q_local = generateVectors( rank )

    particleDim, Nparticles = q_local.shape
    #for now we collect only to root
    root = 0
    #receive counts - assuming flattened arrays
    sendbuf = w_local
    recvcounts = np.array([1, 3, 2], dtype=int)
    sendcount = 2
    sendcounts = np.ones(3) * len(w_local)
    senddisplacements = np.array([0, 1, 2])
    recvdisplacements = np.array([0, 1, 3])
    recvbuf = np.empty( np.sum(recvcounts), dtype=float )

    comm.Gatherv(sendbuf=(sendbuf, recvcounts[rank], MPI.DOUBLE), recvbuf=(recvbuf, recvcounts, MPI.DOUBLE), root=root)
    if rank == root:
        print("Gathered array: ", recvbuf)
    return


def test_gathervVector():
    """
    Simple test of MPI Gatherv with multidimensional flattened arrays as inputs.

    """
    #generate the data
    q_local = generateVectors( rank )
    particleDim, Nparticles = q_local.shape

    #for now we collect only to root
    root = 0
    #receive counts of particles
    recvParticleCounts = np.array([2, 0, 3])
    # copy the first particles of a process 
    # to the buffer and flatten buffer using Fortran order (col-major)
    sendbuf = np.copy( q_local[:, :recvParticleCounts[rank]] )
    sendbuf = sendbuf.flatten(order='F')
    # recvCounts in numbers of elementsof the fundamental datatype
    recvcounts = recvParticleCounts * particleDim
    recvbuf = np.empty( np.sum(recvcounts), dtype=float )
    # Perform the gather operation
    comm.Gatherv(sendbuf=(sendbuf, recvcounts[rank], MPI.DOUBLE), recvbuf=(recvbuf, recvcounts, MPI.DOUBLE), root=root)

    if rank == root:
        print("Sendbuf (pre-reshaping):\n ",  q_local[:, :recvParticleCounts[rank]])
        print("Sendbuf (post-reshaping):\n ", sendbuf)
        print("Recvcounts: ", recvcounts)
        print("Gathered array: \n", recvbuf)
        #reshape into proper shape
        receivedVectors = recvbuf.reshape( (np.sum(recvParticleCounts), particleDim) )
        print("Gathered vectors: \n", receivedVectors.T)
        #generate reference:
        q_3 = generateVectors( 2 )
        referenceSolution = np.zeros( (particleDim, np.sum(recvParticleCounts) ) )
        referenceSolution[:, :2] = q_local[:,:2]
        referenceSolution[:, 2:] = q_3[:,:3]
        print("Reference solution:\n ", referenceSolution)
        assert np.allclose(receivedVectors.T, referenceSolution), "The received data does not match the reference solution!"

    return
 
if __name__=='__main__':
    platform = "cpu"
    numpyro.set_platform(platform)
    if rank == 0:
        print("Testing Gatherv for vectors")
    test_gathervVector();
