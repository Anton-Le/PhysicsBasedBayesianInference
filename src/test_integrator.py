#!/usr/bin/env python3 

from ensemble import Ensemble
from integrator import Leapfrog
import numpy as np

def harmonicPotential(q):
	return q

ensemble = Ensemble(100, 4)

leapfrog = Leapfrog(ensemble, 0.01, 1000, harmonicPotential, 0.01)

leapfrog.getAccel()

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


