#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 30.11.2022

@author: anton

@description:
        Implementations of an empirical step size (dt)
        determination.
"""

import numpy as np
import jax.numpy as jnp 

from jax import grad, vmap, jit 
import jax.lax

from scipy.stats import norm
from scipy.constants import Boltzmann as boltzmannConst
from ensemble import Ensemble
from integrator import Leapfrog, StormerVerlet
import jax 
from functools import partial
import os

def emptyFn(x):
    return
def whileBodyFn(dt, scalingFactor, integrator, potential, q, p, mass, E0):
    dt *= scalingFactor;
    integrator.setStepSize(dt);
    qNew, pNew = integrator.integrate(q,p, mass);
    E1 = 0.5 * jnp.dot(pNew, pNew) / mass + potential(qNew)
    dE = E1 - E0
    return dE

def dtProposalKernel(q, p, mass, potential, T=1.0, dt0=1.0, p_accept = 0.8):
    '''
    This function determines a step size dt for one particle.
    '''
    #print(q, p, mass)
    dt = dt0
    # compute the threshold
    dE_threshold = np.log(1.0/p_accept)*boltzmannConst*T;
    # initialize the integrator
    integrator = Leapfrog(dt, 1*dt, grad(potential) )
    # compute the energy at the initial phase space point
    E0 = 0.5 * jnp.dot(p, p) / mass + potential(q)
    #print("E_0: ", E0.val)
    # Decide whether to magnify or reduce the step size based on
    # the energy difference of current and new configuration.
    integrator.setStepSize(dt)
    qNew, pNew = integrator.integrate(q, p, mass)
    E1 = 0.5 * jnp.dot(pNew, pNew) / mass + potential(qNew)
    #print(E1.val)
    dE = E1 - E0
    #print(dE.val)
    #scalingFactor = 2.0 - (dE < dE_threshold)*3/2
    scalingFactor = 1.0
    scalingFactor = jax.lax.cond( dE < dE_threshold,
                                 lambda x: 2.0,
                                 lambda x: 0.5,
                                 scalingFactor )
    #print("Scaling factor: ", scalingFactor.val)
    cond_fn = lambda x: (x[0] < dE_threshold);
    body_fn = lambda x: (whileBodyFn(x[1], scalingFactor, integrator, potential, q, p, mass, E0), x[1]*scalingFactor)
    dE, dt = jax.lax.while_loop(cond_fn, body_fn, (dE, dt))
    #print(dE.val)
    #print(dt.val)
    # create the termination function
    #while(dE < dE_threshold):
    #        dt *= scalingFactor;
    #        integrator.setStepSize(dt);
    #        qNew, pNew = integrator.integrate(q,p, mass);
    #        E1 = 0.5 * jnp.dot(pNew, pNew) / mass + potential(qNew)
    #        dE = E1 - E0
    #except:
    #    print("Error encountered during the search for the step size")
    #    dt = dt0
    return dt



def dtProposal(ensemble: Ensemble, potential, integrator="Leapfrog"):
    '''
    Function to compute an average step size for a given integrator
    for the entire ensemble s.t. a move is accepted with a given
    probability.
    '''
    #apply dt kernel for each particle
    vectorizedProposal = vmap(dtProposalKernel, in_axes=(0, 0, 0, None, None, None, None), out_axes=0 )
    dt = vectorizedProposal(ensemble.q, ensemble.p, ensemble.mass, potential, ensemble.temperature, 1.0, 0.5)
    return jnp.average(dt)
