#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 2022

@author: bruno
"""

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
