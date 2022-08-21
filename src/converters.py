#!/usr/bin/env python3

"""
@date: 10.08.2022
@author: Anton Lebedev

Implementation of a converter of arrays
and dictionaries.

"""
import numpyro
import jax 
import numpy as np
from collections import OrderedDict
import jax.numpy as jnp

class Converter:
    def __init__(self, model, modelArguments : tuple, modelData : dict):
        """
        Initialize the data
        """
        # run the trace to collect the mappings
        with numpyro.handlers.seed(rng_seed=1):
            trace = numpyro.handlers.trace(model).get_trace(
                            *modelArguments, **modelData
                            )

        self.parametersAndShapes = OrderedDict()

        for paramName in trace.keys():
            paramProperties = trace[paramName]
            if (paramProperties['type'] == 'sample') and (paramProperties['is_observed'] is False):
                self.parametersAndShapes[paramName] = paramProperties['value'].size
        # determine the total length of a vector to store the parameters
        self.vectorSize = 0;
        for item in self.parametersAndShapes.values():
            self.vectorSize += item;

    def toDict(self, parameterVector : np.array):
        """
        Function to convert a position vector (array) into a dictionary
        """
        #ensure shapes check out
        assert self.vectorSize == parameterVector.size, "Vector size incorrect for the number of parameters"
        paramData = OrderedDict().fromkeys( self.parametersAndShapes.keys() );
        # stupid way to distribute chunks of the vector to a dictionary
        arrayIdx = 0
        for paramName in self.parametersAndShapes.keys():
            paramData[paramName] = parameterVector[arrayIdx:arrayIdx + self.parametersAndShapes[paramName] ] #.copy()
            arrayIdx += self.parametersAndShapes[paramName]
        return paramData
    
    def toArray(self, parameterDictionary : dict):
        """
        Function that will convert a dictionary of parameters
        back into an array
        """
        # vec = np.zeros( self.vectorSize )
        vec = jnp.zeros( self.vectorSize )

        arrayIdx = 0
        for paramName in self.parametersAndShapes.keys():
            vec = vec.at[arrayIdx:(arrayIdx + self.parametersAndShapes[paramName]) ].set(
            jnp.ravel(parameterDictionary[paramName]) # ravel is needed to avoid error
            )
            arrayIdx += self.parametersAndShapes[paramName]
        return jnp.array(vec)
