#!/usr/bin/env python3 
import sys
  
# setting path
sys.path.append('../')


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.constants import k as boltzmannConst
from scipy.stats import norm
from ensemble import Ensemble
from potential import noPotential


def boltzmannDistribution(velocity, temperature, mass):
    ''' Velocity boltzman distribution (for magnitude of velocity)
    https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution#Distribution_function
    '''
    mOver2kt = mass / (2 * boltzmannConst * temperature)
    prefactor1 = (mOver2kt / np.pi) ** (3/2)
    prefactor2 = 4 * np.pi * velocity ** 2
    return prefactor1 * prefactor2 * np.exp(- velocity ** 2 * mOver2kt)

def main( ):
    numDimensions = 4
    numParticles = 100

    ensemble1 = Ensemble(numDimensions, numParticles, noPotential)

    # expected output
    qExp = np.zeros( numDimensions )
    pExp = np.zeros( numDimensions )
    mExp = 0
    wExp = 0

    q1, p1, m1, w1 = ensemble1.particle( 10 )

    if (qExp == q1).all( ) and (pExp == p1).all( ) and (mExp == m1).all( ) and \
        (wExp == w1).all( ):
        print('Test 1 passed.')

    try:
        _ = ensemble1.particle(numParticles + 1)
        print('Test 2 Failed')
    except IndexError as error: 
        print(error)
        print('Test 2 Passed \n')



    print('Testing initial velocities follow boltzman distribution.')

    numDimensions2 = 3
    numParticles2 = 1000
    temperature2 = 300
    constMass = 1e-27
    mass2 = np.ones(numParticles2) * constMass
    boltzmannDist2 = lambda velocity: boltzmannDistribution(
        velocity, temperature2, constMass)

    ensemble2 = Ensemble(numDimensions2, numParticles2, noPotential)
    ensemble2.initializeThermal(mass2, temperature2, 3)


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter3D(*ensemble2.q) # check q
    # ax.scatter3D(*(ensemble2.p)) # check p
    # plt.show()
    momentum = ensemble2.p
    momentumMagnitudes = np.linalg.norm(momentum, axis=0)
    velocityMagnitudes = momentumMagnitudes / mass2
    vLinspace = np.linspace(0, max(velocityMagnitudes), 100)
    freq = boltzmannDist2(vLinspace)

    plt.hist(velocityMagnitudes, bins=30, density=True)
    plt.plot(vLinspace, freq)
    plt.show()





if __name__ == '__main__':
    main()
