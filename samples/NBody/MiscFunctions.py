import numpy as np

from matplotlib import pyplot as plt

import sys

#data acquisiton routine
def ReadInput(fname):
	datafile = open(fname, 'r') #open

	row = datafile.readline() #read first line
	rowVars = row.split() #split first line
	N = int(rowVars[0])
	tmax = float(rowVars[1])
	dt = float(rowVars[2])

	#create arrays
	M = np.zeros(N)
	PScoord = np.zeros( (N,2,3) )

	#read masses
	for j in range(N):
		row = datafile.readline()
		M[j] = float(row)

	#read positions
	for j in range(N):
		row = datafile.readline()
		rowVars = row.split()
		PScoord[j,0,0]= float(rowVars[0])
		PScoord[j,0,1]= float(rowVars[1])
		PScoord[j,0,2]= float(rowVars[2])
	
	#read velocities
	for j in range(N):
		row=datafile.readline()
		rowVars=row.split()
		PScoord[j,1,0]= float(rowVars[0] )
		PScoord[j,1,1]= float(rowVars[1] )
		PScoord[j,1,2]= float(rowVars[2] )
	
	#return
	return [N, tmax, dt, M, PScoord]

###################################################################

###################################################################
#define a function to compute the accelerations
###################################################################
def Accel(r, M, G):
	'''assumption r_i = [rx,ry,rz]_i
	are positions in the CoM system
	'''
	a = np.zeros_like(r) # create the array
	N = np.shape(r)[0] #total numer of particles
	#now compute in the dumbest (<=> expensive)
	#way possible

	for i in range( N//2 ):
		r0 = r[i,:]
		dr = r - r0;
		drNrm = np.linalg.norm(dr, axis=1) #row-by-row norm
		da = -1*M[i] * dr / drNrm[:,None]**3
		#mask NAN
		mask = np.where(np.isnan(da))
		da[mask] = 0.0
		# update all elements except the i-th
		a += da;
		#update the i-th element
		da = dr * M[:,None]/drNrm[:,None]**3
		#mask NAN
		mask = np.where(np.isnan(da))
		da[mask] = 0.0
		a[i,:] += np.sum(da, axis=0)
	#scale by gravity constant
	a *= G;
	return a

###################################################################

###################################################################
#define a function to compute the derivative of acceleration by
#modified eq. 1.4
###################################################################
#def AccelDeriv(r_arg, v_arg, M_arg, G):
#	r=np.copy(r_arg)
#	v=np.copy(v_arg)
#	M=np.copy(M_arg)
#
#	#create the appropriate vector
#	da = np.zeros_like(r)
#
#	N = np.shape(r_arg)[0]
#	for i in range( N//2 ):
#		r0=r[i,:]
#		v0=v[i,:]
#		dr = r - r0
#		dv = v - v0
#		drNrm = np.linalg.norm(dr, axis=1) #row-by-row norm
#		#row-row dot
#		tmp = np.einsum('ij,ij->i', dr,dv)
#		increment = dv/drNrm[:,None]**3 - 3*dr*tmp[:,None]/drNrm[:,None]**5
#		mask = np.where(np.isnan(increment))[0]
#		increment[mask] = 0.0;
#		da -= M[i] *increment
#		#update the ith element
#		da[i,:] += np.sum(increment * M[:,None], axis=0)
#	da *= G;
#	return da


###################################################################
#define a function to compute the derivative of acceleration by
#modified eq. 1.4
###################################################################
def AccelDeriv(r_arg, v_arg, M_arg, G):
	r=np.copy(r_arg)
	v=np.copy(v_arg)
	M=np.copy(M_arg)

	#create the appropriate vector
	da = np.zeros_like(r)

	N = np.shape(r_arg)[0]
	for i in range( N//2 ):
		r0=r[i,:]
		v0=v[i,:]
		for j in range(N):
			if i==j:
				continue
			else:
				dr = r[j,:] - r0
				dv = v[j,:] - v0
				drNrm = np.linalg.norm(dr)
				da[i,:] += G*M[j] * (dv/drNrm**3 - 3*dr * np.dot(dv,dr)/drNrm**5 )
				da[j,:] += G*M[i] * (-1.0)*(dv/drNrm**3 - 3*dr * np.dot(dv,dr)/drNrm**5)
	
	return da




###################################################################
#define a function to compute the total energy
##################################################################
def Energy(r, v, M, G):
	#kinetic energy
	T = 0.0
	N=len(r)
	vNrmSquared = np.linalg.norm(v, axis=1)**2
	T = np.sum(vNrmSquared *M[:,None])
	T *= 0.5;
	
	#potential energy
	U = 0
	for i in range(N):
		r0 = r[i,:]
		for j in range(i+1,N):
			dr=r[j,:]-r0
			U += -1.0*G* M[i] * M[j] / np.linalg.norm(dr)
	
	return T+U


#############################################################
#define a function to adapt the timestep by using
#eq. 1.65 form the manual
#############################################################
def AdaptiveTimeSimple(r_arg, v_arg, M_arg, G, dt0):
	'''{r,v,M}_arg , G Input data required to calculate
	a, da/dt
	dt0 - initial time step - to be refined'''
	r = np.copy(r_arg)
	v = np.copy(v_arg)
	M = np.copy(M_arg)
	a = Accel(r,M,G)
	da = AccelDeriv(r,v,M,G)
	N= np.shape(v)[0]
	#a hardwired ||a_i|| / ||da_i||
	dt = np.sqrt( np.sum( a**2, axis=1 ) ) / np.sqrt( np.sum( da**2, axis=1 ) )
	
#	minFac = 3.0
#	for j in range(N):
#	  fac = np.linalg.norm(a[j,:]) / np.linalg.norm( da[j,:] )
#	  if fac< minFac:
#	    minFac=fac
	
	#dtmin = dt0*minFac
	dtmin = dt0 * min(dt)
	print("dtmin=", dtmin)
	return dtmin

#############################################################
#adaptive step for hermite-like integrators (e.g. RK)
#using eq. 1.66 from the lab manual
#NOTE: to not pollute the memory we copy the accelerations
#here.
#############################################################
def AdaptiveTimeAdvanced(r_arg, v_arg, M_arg, G, dt0):
	#copy data
	dt = dt0
	r = np.copy(r_arg)
	v = np.copy(v_arg)
	M = np.copy(M_arg)
	#extra arrays for the prediction step
	vp = np.zeros_like(v)
	rp = np.zeros_like(r)
	ap = np.zeros_like(v)
	dap = np.zeros_like(v)
	#determine the quantities
	#NOTE: implicit deinition of da, a
	a = Accel(r,M,G)
	da = AccelDeriv(r,v,M,G)
	vp = v + dt*a + dt**2 * da/2
	rp = r + dt*v + dt**2 * a/2 + dt**3 * da/6
	#determine the predicted accelerations
	ap = Accel(rp,M,G)
	dap = AccelDeriv(rp,vp,M,G)
	#determine a(2), a(3)
	a2 = -6* (a-ap)/(dt**2) - 2*(2*da+dap)/dt
	a3 = 12* (a-ap)/(dt**3) + 6*(da+dap)/(dt**2)
	#determine the value in the root of the equation
	#1.66
	val = np.sqrt( np.sum( a**2, axis=1) ) * np.sqrt( np.sum(a2**2, axis=1) ) +\
			np.sum( da**2, axis=1) / ( np.sqrt( np.sum(da**2,axis=1) ) *\
			np.sqrt( np.sum(a3**2, axis=1) ) +np.sum(a2**2,axis=1) )

	dtmin = dt0 * min( np.sqrt(val) )

	return dtmin

#############################################################
#define a function for plotting
#############################################################
def Graphical(r, v, Dir=".", Fname="plot"):

	fig = plt.figure()
	ax=fig.gca(projection="3d")

	x = r[:,0]
	y = r[:,1]
	z = r[:,2]

	ax.plot(x,y,z)
	ax.set_xlabel("X Axis")
	ax.set_ylabel("Y Axis")
	ax.set_zlabel("Z Axis")
	Filename=Dir + Fname+".png"
	plt.savefig(Fname)
	plt.close()
	plt.clf()
