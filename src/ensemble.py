import numpy as np

class Ensemble( ):


    def __init__(self, numParticles, numDimensions):
        
        self.numParticles = numParticles
        self.q = np.zeros((numParticles, numDimensions))
        self.p = np.zeros((numParticles, numDimensions))
        self.m = np.zeros(numParticles)
        self.weights = np.zeros(numParticles)


    def particle(self, particleNum):
# I'm not sure if brackets follow correct style here - please correct if needed        
        if not 0 <= particleNum < self.numParticles:

            raise IndexError(f'Index {particleNum} out of bounds. '\
                f'numParticles={self.numParticles}') 

        return self.q[particleNum], self.p[particleNum], self.m[particleNum], \
            self.weights[particleNum]
