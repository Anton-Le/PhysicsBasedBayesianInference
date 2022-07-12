from ensemble import Ensemble
from integrator import Leapfrog
import numpy as np

def harmonicPotential(q):
	return q

ensemble = Ensemble(100, 4)

leapfrog = Leapfrog(ensemble, 0.01, 1000, harmonicPotential, 0.01)

leapfrog.getAccel()

