import numpy as np
from scipy.optimize import fsolve

import sys, time
from matplotlib import pyplot as plt

from MiscFunctions import Accel, AccelDeriv, Energy
from MiscFunctions import AdaptiveTimeSimple, AdaptiveTimeAdvanced, Graphical

from ode45 import ode45

###################################################################
# define explicit euler solver
###################################################################
def IntegrateEE(r_arg, v_arg, M_arg, G=1, tmax=100, dt=0.1, flag=False):
    """Assumption: r = [rx,ry,rz]
    v = [vx,vy,vz]
    are coordinates and velocities w.r.t. the CoM
    """
    Ntot = int(tmax / dt)  # total amount of timesteps
    En = np.zeros(Ntot)

    r = np.copy(r_arg)
    v = np.copy(v_arg)
    M = np.copy(M_arg)

    for n in range(Ntot):
        a = Accel(r, M, G)
        r += dt * v
        v += dt * a
        En[n] = Energy(r, v, M, G)

    if flag:
        return [r, v, En]
    else:
        return [r, v, En[-1]]


####################################################################
# define the second euler solver
####################################################################


def IntegrateEC(r_arg, v_arg, M_arg, G=1, tmax=100, dt=0.1, flag=False):
    """Assumption: r = [rx,ry,rz]
    v = [vx,vy,vz]
    are coordinates and velocities w.r.t. the CoM
    """
    Ntot = int(tmax / dt)  # total amount of timesteps
    En = np.zeros(Ntot)

    r = np.copy(r_arg)
    v = np.copy(v_arg)
    M = np.copy(M_arg)

    for n in range(Ntot):
        a = Accel(r, M, G)
        v += dt * a
        r += dt * v
        En[n] = Energy(r, v, M, G)

    if flag:
        return [r, v, En]
    else:
        return [r, v, En[-1]]


###################################################################
# define mv-solver
###################################################################
def IntegrateMV(r_arg, v_arg, M_arg, G=1, tmax=100, dt=0.1, flag=False):
    """Assumption: r = [rx,ry,rz]
    v = [vx,vy,vz]
    are coordinates and velocities w.r.t. the CoM
    """
    Ntot = int(tmax / dt)  # total amount of timesteps
    En = np.zeros(Ntot)

    r = np.copy(r_arg)
    v = np.copy(v_arg)
    M = np.copy(M_arg)

    for n in range(Ntot):
        a = Accel(r, M, G)
        r += 0.5 * dt * v
        v += dt * a
        r += 0.5 * dt * v
        En[n] = Energy(r, v, M, G)

    if flag:
        return [r, v, En]
    else:
        return [r, v, En[-1]]


##################################################################
# define the Leapfrog integrator
##################################################################
def IntegrateLF(r_arg, v_arg, M_arg, G=1, tmax=100, dt=0.1, flag=False):
    # copy data to local data
    r = np.copy(r_arg)
    v = np.copy(v_arg)
    M = np.copy(M_arg)
    # determine the number of steps to be taken
    Ntot = int(tmax / dt)
    # initial step
    r += v * dt / 2.0
    En = np.zeros(Ntot)
    vInterm = np.zeros_like(v)
    for t in range(Ntot):
        a = Accel(r, M, G)  # includes initial step
        # determine intermediate velocity to determine the energy
        vInterm = (2.0 * v + dt * a) / 2.0

        v += dt * a
        r += dt * v
        En[t] = Energy(r, vInterm, M, G)

    # final timestep
    r += dt * v / 2.0
    if flag:
        return [r, v, En]
    else:
        return [r, v, En[-1]]


####################################################################
# define the Verlet integrator
###################################################################
def IntegrateVerlet(r_arg, v_arg, M_arg, G=1, tmax=100, dt=0.1, flag=False):
    # copy data
    r = np.copy(r_arg)
    v = np.copy(v_arg)
    M = np.copy(M_arg)
    Ntot = int(tmax / dt)
    # additional vectors for positions and velocities
    v2 = np.zeros_like(v)
    r2 = np.zeros_like(r)
    # first step
    a = Accel(r, M, G)
    r2 = r - dt * v + 0.5 * a * dt**2
    En = np.zeros(Ntot)

    for t in range(Ntot):
        a = Accel(r, M, G)
        rtmp = 2 * r - r2 + a * dt**2
        v = (rtmp - r2) / (2 * dt)
        En[t] = Energy(r, v, M, G)
        r2 = r
        r = rtmp

    # final timestep#
    a = Accel(r, M, G)
    r = 2 * r - r2 + a * dt**2
    v = (r - r2) / (2 * dt)

    if flag:
        return [r, v, En]
    else:
        return [r, v, En[-1]]


####################################################################
# define the Stromer-Verlet algorithm
####################################################################
def IntegrateSV(r_arg, v_arg, M_arg, G=1, tmax=100, dt=0.1, flag=False):
    r = np.copy(r_arg)
    v = np.copy(v_arg)
    M = np.copy(M_arg)
    Ntot = int(tmax / dt)
    En = np.zeros(Ntot)

    for t in range(Ntot):
        a = Accel(r, M, G)
        v += 0.5 * dt * a
        r += dt * v
        a = Accel(r, M, G)
        v += 0.5 * dt * a
        En[t] = Energy(r, v, M, G)

    if flag:
        return [r, v, En]
    else:
        return [r, v, En[-1]]


######################################################################
# define Hermite-Integrator
######################################################################
def IntegrateHI(
    r_arg, v_arg, M_arg, G=1, tmax=100, dt=0.1, MaxRef=1, flag=False
):
    r = np.copy(r_arg)
    v = np.copy(v_arg)
    M = np.copy(M_arg)
    Ntot = int(tmax / dt)
    En = np.zeros(Ntot)
    # temporary vectors
    vp = np.zeros_like(v)
    vc = np.zeros_like(v)
    rp = np.zeros_like(r)
    rc = np.zeros_like(r)
    ap = np.zeros_like(v)
    a = np.zeros_like(v)
    dap = np.zeros_like(v)
    # energies
    En = np.zeros(Ntot)
    for t in range(Ntot):
        a = Accel(r, M, G)
        da = AccelDeriv(r, v, M, G)
        vc = v
        rc = r
        vp = v + dt * a + dt**2 * da / 2
        rp = r + dt * v + dt**2 * a / 2 + dt**3 * da / 6
        ap = Accel(rp, M, G)
        dap = AccelDeriv(rp, vp, M, G)
        for i in range(MaxRef):
            vc = v + dt * (ap + a) / 2 + dt**2 * (dap - da) / 12
            rc = r + dt * (vc + v) / 2 + dt**2 * (ap - a) / 12
            ap = Accel(rc, M, G)
            dap = AccelDeriv(rc, vc, M, G)

        # vp = v+dt*(ap+a)/2 + dt**2*(dap-da)/12
        # rp = r + dt*(vp+v)/2 + dt**2*(ap-a)/12
        v = vc
        r = rc
        En[t] = Energy(r, v, M, G)

    if flag:
        return [r, v, En]
    else:
        return [r, v, En[-1]]


####################################################################
# define the Stromer-Verlet algorithm - with adaptive time
####################################################################
def IntegrateSVA(r_arg, v_arg, M_arg, G=1, tmax=100, dt=0.1, flag=False):
    r = np.copy(r_arg)
    v = np.copy(v_arg)
    M = np.copy(M_arg)
    dt0 = dt
    # create a list of times and energies
    T = []
    En = []

    t = 0.0
    while t <= tmax:
        a = Accel(r, M, G)
        # determine new timestep length
        # dt = AdaptiveTimeSimple(r,v,M, G, dt0)
        dt = AdaptiveTimeAdvanced(r, v, M, G, dt0)
        v += 0.5 * dt * a
        r += dt * v
        a = Accel(r, M, G)
        v += 0.5 * dt * a

        t += dt
        T.append(t)
        en = Energy(r, v, M, G)
        En.append(en)

    if flag:
        return [r, v, T, En]
    else:
        return [r, v, En[-1]]


###################################################################
# define explicit euler solver - with adaptive time
###################################################################
def IntegrateEEA(r_arg, v_arg, M_arg, G=1, tmax=100, dt=0.1, flag=False):
    """Assumption: r = [rx,ry,rz]
    v = [vx,vy,vz]
    are coordinates and velocities w.r.t. the CoM
    """
    r = np.copy(r_arg)
    v = np.copy(v_arg)
    M = np.copy(M_arg)
    dt0 = dt

    T = []
    En = []
    t = 0.0
    while t <= tmax:
        a = Accel(r, M, G)
        dt = AdaptiveTimeSimple(r, v, M, G, dt0)
        r += dt * v
        v += dt * a
        en = Energy(r, v, M, G)
        t += dt
        T.append(t)
        En.append(en)

    if flag:
        return [r, v, T, En]
    else:
        return [r, v, En[-1]]


######################################################################
# define Hermite-Integrator - adaptive time
######################################################################
def IntegrateHIA(
    r_arg, v_arg, M_arg, G=1, tmax=100, dt=0.1, MaxRef=1, flag=False
):
    r = np.copy(r_arg)
    v = np.copy(v_arg)
    M = np.copy(M_arg)
    dt0 = dt
    # temporary vectors
    vp = np.zeros_like(v)
    vc = np.zeros_like(v)
    rp = np.zeros_like(r)
    rc = np.zeros_like(r)
    ap = np.zeros_like(v)
    a = np.zeros_like(v)
    dap = np.zeros_like(v)
    # log
    T = []
    En = []
    t = 0.0
    while t <= tmax:
        a = Accel(r, M, G)
        da = AccelDeriv(r, v, M, G)
        dt = AdaptiveTimeAdvanced(r, v, M, G, dt0)
        vc = v
        rc = r
        vp = v + dt * a + dt**2 * da / 2
        rp = r + dt * v + dt**2 * a / 2 + dt**3 * da / 6
        ap = Accel(rp, M, G)
        dap = AccelDeriv(rp, vp, M, G)
        for i in range(MaxRef):
            vc = v + dt * (ap + a) / 2 + dt**2 * (dap - da) / 12
            rc = r + dt * (vc + v) / 2 + dt**2 * (ap - a) / 12
            ap = Accel(rc, M, G)
            dap = AccelDeriv(rc, vc, M, G)
        v = vc
        r = rc
        t += dt
        T.append(t)
        En.append(Energy(r, v, M, G))
    if flag:
        return [r, v, T, En]
    else:
        return [r, v, En[-1]]


##########################################################
# ODE for the N-Body problem
##########################################################
def NBodyODE(y, M, G, N):
    r = np.copy(y[: 3 * N])
    v = y[3 * N :]
    # 	print "y=",y

    r0 = np.reshape(r, (N, 3))

    a = Accel(r0, M, G)
    # da = AccelDeriv(y[:Npart, :], y[Npart:, :] , M, G)
    # dydt = np.zeros_like(y)

    # dydt[0:N,:]= y[0:N,:]
    # dydt[N:,:] = a

    dydt = np.array([v, a.flatten()])
    return dydt.flatten()


##########################################################
# define RK45-Integrator
##########################################################
def IntegrateRK(r_arg, v_arg, M, G, tmax, dt, flag=False):
    y0 = (np.array([r_arg.flatten(), v_arg.flatten()])).flatten()
    tspan = (0, tmax)
    N = np.shape(r_arg)[0]
    t, y = ode45(lambda t, y: NBodyODE(y, M, G, N), tspan, y0)
    if flag:
        return [t, y]
    else:
        return [t, y[-1][0]]


##########################################################
# implicit midpoint rule
##########################################################
def IntegrateIMP(r, v, M, G, tmax, dt, flag=False):
    N = np.shape(r)[0]
    # 	print "Npart=", N
    y0 = (np.array([r.flatten(), v.flatten()])).flatten()
    NSteps = int(tmax / dt)

    y = np.zeros((NSteps + 1, 6 * N))

    t = np.zeros(NSteps + 1)
    y[0, :] = y0
    for i in range(NSteps):
        t[i + 1] = t[i] + dt
        sol = fsolve(
            lambda p: y[i, :] + dt * NBodyODE((y[i, :] + p) / 2.0, M, G, N) - p,
            y[i + 1, :],
        )
        y[i + 1, :] = sol
    if flag:
        return (t, y)
    else:
        return [t, y[-1][0]]
