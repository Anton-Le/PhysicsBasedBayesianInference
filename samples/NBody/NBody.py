import numpy as np
import scipy as sp

import time

from matplotlib import pyplot as plt

from Integrators import IntegrateEC, IntegrateSV, IntegrateVerlet, IntegrateMV, IntegrateLF
from Integrators import IntegrateHI, IntegrateSVA, IntegrateEEA, IntegrateHIA, IntegrateRK, IntegrateIMP
from Integrators2B import *
from MiscFunctions import ReadInput, Accel, Energy




###################################################################
#Main programm branch

if __name__ == '__main__':
	Filename = "in2.txt"
	#Filename = "pl.100"
	N, tmax, dt, M, PScoord = ReadInput(Filename)
	print("N: ", N, "tmax: ", tmax, "dt: ", dt)
	print("Masses: ")
	print(M)

	#normalize mass
	M = M/np.linalg.norm(M)
	#set the universal gravitational constant
	G = 1.0
	#calculate the center of mass
	Rcom = np.sum( M* np.transpose(PScoord[:,0,:]), axis=1) / np.sum(M)
	print(Rcom)
	#calculate the velocity of the CoM
	Vcom = np.sum(M * np.transpose( PScoord[:,1,:] ),axis=1) / np.sum(M)
	print(Vcom)
	
	#transform into the CoM system
	PScoord[:,0,:] -= Rcom
	PScoord[:,1,:] -= Vcom

	#determine the forces
	a = Accel(PScoord[:,0,:], M, G)
	F = M * np.transpose(a)

	#determine the initial Energy
	Estart = Energy(PScoord[:,0,:], PScoord[:,1,:], M, G)

	tstart=time.time()
	rSV, vSV, EnSV, L, R, AE = ISV2B(PScoord[:,0,:], PScoord[:,1,:], M, G, tmax, dt, True)
	rtSV = time.time()- tstart
	Ntot = int( tmax/dt )
	t = np.array(range(Ntot)) * dt

	ySV = np.log10( abs(EnSV- Estart) )
	LSV = np.log10( abs(L - L[0]) )
	RSV = np.log10( abs(R - R[0]) )
#	#plot energy difference

	plt.plot(t,ySV, 'k--', t, LSV, 'y-', t, RSV, 'g-' )
	plt.xlabel("time [arb. units]")
	plt.ylabel("$\log |E_t - E_0|$")
	plt.title("Energy difference evolution for Stromer-Verlet")
	plt.show()
	plt.savefig("SV_100.png")
	
	#Hermite
	r=PScoord[:,0,:]
	v=PScoord[:,1,:]
	tstart = time.time()
	rHI, vHI, EnHI, L, R, AE = IHI2B(r, v, M, G, tmax, dt, flag=True)
	rtHI = time.time() - tstart
	
	yHI = np.log10( abs( EnHI - Estart) )
	plt.clf()
	plt.plot(t, yHI, 'b-')
	plt.title("Energy evolution adaptive-time HI - 100 particles")
	plt.xlabel("time [arb. unit]")
	plt.ylabel("$\log(|E_t - E_0 |)$")
	plt.show()
	plt.savefig("HI_100.png")
	
	#adaptive time tests
	r=PScoord[:,0,:]
	v=PScoord[:,1,:]
	
	tmax = 1.0
	dt=0.5
	#runge-kutta

	#adaptive-time stromer-verlet
	tstart = time.time()
	rSVA, vSVA, tSVA, EnSVA = IntegrateSVA(r, v, M, G, tmax, dt, True)
	rtSVA = time.time() - tstart

	ySVA = np.log10( abs( EnSVA - Estart) )
	plt.clf()
	plt.plot(tSVA, ySVA, 'b-')
	plt.title("Energy evolution adaptive-time SV - 100 particles")
	plt.xlabel("time [arb. unit]")
	plt.ylabel("$\log(|E_t - E_0 |)$")
	plt.show()
	plt.savefig("SVA_100.png")


	print("runtime SV: ", rtSV)
	print("runtime SVA: ", rtSVA)
	print("runtime HI: ", rtHI)
