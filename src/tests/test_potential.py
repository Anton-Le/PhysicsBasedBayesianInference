import sys
  
# setting path
sys.path.append('../')

from potential import harmonicPotentialND, getForce
from ensemble import Ensemble
import numpy as np



springConsts = np.array([2, 3])
harmonicPotential1 = lambda q: harmonicPotentialND(q, springConsts) 

ensemble1 = Ensemble(2, 10, harmonicPotential1)
ensemble1.mass = np.ones(10)

print('We expect a potential of 2*3^2 + 3*4^2 / 2 =18+48 /2=33 at (3, 4)')
a = np.array([3, 4])
ensemble1.q[:, 0] = a
print(ensemble1.q)
expPot = 33
calculatedPot = harmonicPotential1(ensemble1.q)[0]
print('Calculated potential at (3, 4) with springConsts (2, 3)')
if calculatedPot == expPot:
	print(f'={expPot}')
	print('Pass')

# print(harmonicPotential1(a))

# We expect the force to be (6, 12)
print(ensemble1.getAccel())
print('getAccel works')
