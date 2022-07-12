
import numpy as np

class Integrator():
    def integrationMethod():
        raise NotImplementedError('Integrator superclass does not specify \
            integration method')
    
    def plottingIntegrationMethod():
        raise NotImplementedError('Method not defined.')


    def __init__(self, ensemble, dt, numSteps, potential, dq):
        self.q, self.p, self.mass, self.weights = ensemble
        self.potential = potential
        self.dq = dq


    def getAccel(self):
        plus = self.potential(self.q + self.dq)
        minus = self.potential(self.q - self.dq)
        force = (minus - plus) / (2 * self.dq)
        return force / self.mass


class Leapfrog(Integrator):


    def integrationMethod(self):
        v = self.p / self.mass 
        aCurrent = self.getAccel()
        
        for i in range(numSteps):
            self.ensemble.x += v * dt + 0.5 * a_current * dt ** 2
            
            aNext = self.getAccel

            v += 0.5 * (aNext + aCurrent)


            aCurrent = aNext

        self.p = v * self.mass 

        return self.q, self.p

    def plottingIntegrationMethod(self):
        return 0


