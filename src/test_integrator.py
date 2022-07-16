
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 2022

@author: bruno
"""
=======

from ensemble import Ensemble
from integrator import integrator_methods
import numpy as np

def harmonicPotential(q):
    return q

########################
def main():
    # integrator setup
    finalTime = 1
    stepSize = 0.01
    method = 'SV'
    # ensemble setup
    numDimensions = 100
    numParticles = 100
    mass = np.ones(numParticles)
    temperature = 1
    q_std = 0.1
    
    #ensemble initialization
    ensemble1 = Ensemble(numDimensions, numParticles, harmonicPotential)
    ensemble1.initializeThermal(mass, temperature, q_std)
    
    # object of class integartor
    sol_p_q = integrator_methods(ensemble1, stepSize, finalTime)
    
    # actual solution for position and momenta
    p, q = sol_p_q.numerical_solution(method)
    return p, q
        
if __name__ == '__main__':
    p, q = main()
=======
def freeParticleAnalytic(ensemble, numSteps, dt):
	time = numSteps * dt
	q = ensemble.q * time * ensemble.p / ensemble.mass

	return q, ensemble.p

def harmonicOscillatorAnalytic(ensemble, numSteps, dt, springConsts):
    omega = sqrt(springConsts / mass)
    initialV = ensemble.p / ensemble.mass
    # constants
    a = 0.5 * (ensemble.q - (1j / omega) * initialV)
    b = ensemble.q - a
    
    q = np.real(a * np.exp(1j * omega * time) + b * np.exp(-1j * omega * time))
    
    v = np.real(1j * omega * (a * np.exp(1j * omega * time) - b * np.exp(-1j * omega * time)))

	return q, v * ensemble.mass

