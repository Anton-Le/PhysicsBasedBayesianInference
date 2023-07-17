#!/usr/bin/env python3
# This file contains simple tests of MPI 
# operations required to parallelise SMC
#

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

def test_allgather():
    """
    This function is used to gather 
    """
    assert size == 3, "The MPI tests require 3 processes!"
    #generate a uniform distribution of weights, 10 each
    #modified by the rank
    weights = generateWeights( rank )

    globalWeights, _ = mpi4jax.allgather(weights)
    globalWeights = globalWeights.flatten()
    cdf = np.cumsum(globalWeights / np.sum(globalWeights) )
    referenceCDF = generateReferenceCDF()
    for i in range(size):
        if rank == i:
                print("Process ", i, "w = ", weights, "global : ", globalWeights)
                print("CDF: ", cdf)
    assert np.allclose(cdf, referenceCDF), "The CDFs do not match!"
    return

def test_scan():
    """
    This function tests the MPI_Scan application.
    It is equivalent to an (exclusive) prefix sum and simplifies
    the calculation of a cumulative distribution function.
    """
    assert size == 3, "The MPI tests require 3 processes!"
    #generate a uniform distribution of weights, 10 each
    #modified by the rank
    NParticlesLocal = 10
    weights = np.ones( NParticlesLocal )
    weights *= (size - rank)

    NParticlesTotal = NParticlesLocal * size

    cdf, _ = mpi4jax.scan(weights, mpi4py.MPI.SUM)
    for i in range(size):
        if rank == i:
                print("Process ", i, "w = ", weights, "cdf : ", cdf)


def test_gatherv_simple():
    """
    Test of the MPI Gatherv function for the collection of
    a subset of positions and momenta from the workers.
    Since MPI4JAX does not provide this function we
    need to fall back to MPI4Py.
    """
    #generate the data
    root = 0
    local_array = [rank] * random.randint(2, 7)
    print("rank: {}, local_array: {}".format(rank, local_array))
    sendbuf = np.array(local_array)
    #Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.gather(len(sendbuf), root))
    if rank == root:
        print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
        recvbuf = np.empty(sum(sendcounts), dtype=int)
    else:
        recvbuf = None
    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)
    if rank == root:
            print("Gathered array: {}".format(recvbuf))

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

if __name__=='__main__':
    platform = "cpu"
    numpyro.set_platform(platform)
    if rank == 0:
        print("Testing simple GATHER ")
    test_gather();
    mpi4jax.barrier();
    if rank == 0:
        print("Testing SCAN")
    test_scan();
    mpi4jax.barrier();
    if rank==0:
        print("Testing ALLGATHER")
    test_allgather();
    mpi4jax.barrier();
    if rank == 0:
        print("Testing GATHERV")
    test_gatherv_simple()
    mpi4jax.barrier();
    if rank==0:
        print("Testing GATHERV with vectors")
    test_gatherv();
