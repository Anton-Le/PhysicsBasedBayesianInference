import numpy as np

def harmonicPotentialND(q, k):
	if q.shape[0] != len(k):
		raise ValueError('k must be 1D array corresponding with spring constant \
			in 3 dimensions.')
	return 0.5 * k * q ** 2

def noPotential(q):
	return 0