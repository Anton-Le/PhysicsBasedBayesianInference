import numpy as np
from scipy.optimize import fsolve

from MiscFunctions import Accel, AccelDeriv, Energy
from MiscFunctions import AdaptiveTimeSimple, AdaptiveTimeAdvanced

from ode45 import ode45

###################################################################
# define mv-solver
###################################################################
def IMV2B(r_arg, v_arg, M_arg, G=1, tmax=100, dt=0.1, flag=False):
    """Assumption: r = [rx,ry,rz]
    v = [vx,vy,vz]
    are coordinates and velocities w.r.t. the CoM
    """
    Ntot = int(tmax / dt)  # total amount of timesteps
    En = np.zeros(Ntot)
    Lnrm = np.zeros(Ntot)
    Rnrm = np.zeros(Ntot)
    AE = np.zeros(Ntot)

    r = np.copy(r_arg)
    v = np.copy(v_arg)
    M = np.copy(M_arg)
    for n in range(Ntot):
        a = Accel(r, M, G)
        r += 0.5 * dt * v
        v += dt * a
        r += 0.5 * dt * v
        En[n] = Energy(r, v, M, G)
        relR = r[0, :] - r[1, :]
        relV = v[0, :] - v[1, :]
        L = np.cross(relR, relV) * M[0] * M[1] / (M[0] + M[1])
        R = (
            1.0
            / (G * (M[0] + M[1]))
            * (np.cross(relV, L) - relR / np.linalg.norm(relR))
        )
        Lnrm[n] = np.linalg.norm(L)
        Rnrm[n] = np.linalg.norm(R)
        AE[n] = Lnrm[n] ** 2 / (G * (M[0] + M[1]) * (1 - Rnrm[n] ** 2))

    if flag:
        return [r, v, En, Lnrm, Rnrm, AE]
    else:
        return [r, v, En[-1]]


##################################################################
# define the Leapfrog integrator
##################################################################
def ILF2B(r_arg, v_arg, M_arg, G=1, tmax=100, dt=0.1, flag=False):
    # copy data to local data
    r = np.copy(r_arg)
    v = np.copy(v_arg)
    M = np.copy(M_arg)
    # determine the number of steps to be taken
    Ntot = int(tmax / dt)
    # initial step
    r += v * dt / 2.0
    En = np.zeros(Ntot)
    Lnrm = np.zeros(Ntot)
    Rnrm = np.zeros(Ntot)
    AE = np.zeros(Ntot)
    vInterm = np.zeros_like(v)
    Pos = np.zeros((Ntot, np.shape(r)[0] * 3))
    for t in range(Ntot):
        a = Accel(r, M, G)  # includes initial step
        # determine intermediate velocity to determine the energy
        vInterm = (2.0 * v + dt * a) / 2.0

        v += dt * a
        r += dt * v
        Pos[t, :] = r.flatten()
        En[t] = Energy(r, vInterm, M, G)
        relR = r[0, :] - r[1, :]
        relV = v[0, :] - v[1, :]
        L = np.cross(relR, relV) * M[0] * M[1] / (M[0] + M[1])
        R = (
            1.0
            / (G * (M[0] + M[1]))
            * (np.cross(relV, L) - relR / np.linalg.norm(relR))
        )
        Lnrm[t] = np.linalg.norm(L)
        Rnrm[t] = np.linalg.norm(R)
        AE[t] = Lnrm[t] ** 2 / (G * (M[0] + M[1]) * (1 - Rnrm[t] ** 2))

    # final timestep
    r += dt * v / 2.0
    Pos[-1, :] = r.flatten()
    if flag:
        return [Pos, v, En, Lnrm, Rnrm, AE]
    else:
        return [r, v, En[-1]]


####################################################################
# define the Stromer-Verlet algorithm
####################################################################


def ISV2B(r_arg, v_arg, M_arg, G=1, tmax=100, dt=0.1, flag=False):
    r = np.copy(r_arg)
    v = np.copy(v_arg)
    M = np.copy(M_arg)
    Ntot = int(tmax / dt)
    En = np.zeros(Ntot)
    Lnrm = np.zeros(Ntot)
    Rnrm = np.zeros(Ntot)
    Pos = np.zeros((Ntot, np.shape(r)[0] * 3))
    AE = np.zeros(Ntot)

    for t in range(Ntot):
        a = Accel(r, M, G)
        v += 0.5 * dt * a
        r += dt * v
        a = Accel(r, M, G)
        v += 0.5 * dt * a
        Pos[t, :] = r.flatten()
        En[t] = Energy(r, v, M, G)
        relR = r[0, :] - r[1, :]
        relV = v[0, :] - v[1, :]
        L = np.cross(relR, relV) * M[0] * M[1] / (M[0] + M[1])
        R = (
            1.0
            / (G * (M[0] + M[1]))
            * (np.cross(relV, L) - relR / np.linalg.norm(relR))
        )
        Lnrm[t] = np.linalg.norm(L)
        Rnrm[t] = np.linalg.norm(R)
        AE[t] = Lnrm[t] ** 2 / (G * (M[0] + M[1]) * (1 - Rnrm[t] ** 2))

    if flag:
        return [Pos, v, En, Lnrm, Rnrm, AE]
    else:
        return [r, v, En[-1]]


######################################################################
# define Hermite-Integrator
######################################################################
def IHI2B(r_arg, v_arg, M_arg, G=1, tmax=100, dt=0.1, MaxRef=1, flag=False):
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
    Lnrm = np.zeros(Ntot)
    Rnrm = np.zeros(Ntot)
    AE = np.zeros(Ntot)
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
        relR = r[0, :] - r[1, :]
        relV = v[0, :] - v[1, :]
        L = np.cross(relR, relV) * M[0] * M[1] / (M[0] + M[1])
        R = (
            1.0
            / (G * (M[0] + M[1]))
            * (np.cross(relV, L) - relR / np.linalg.norm(relR))
        )
        Lnrm[t] = np.linalg.norm(L)
        Rnrm[t] = np.linalg.norm(R)
        AE[t] = Lnrm[t] ** 2 / (G * (M[0] + M[1]) * (1 - Rnrm[t] ** 2))

    if flag:
        return [r, v, En, Lnrm, Rnrm, AE]
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
        dt = AdaptiveTimeSimple(r, v, M, G, dt0)
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
# specific 2-Body ODE
##########################################################
def ODE2B(y, M, G):
    """ODE for the relative position vector r=r_1 - r_2"""
    r = np.copy(y[:3])
    v = np.copy(y[3:])
    Mtot = sum(M)
    dydt = np.array([v, -G * Mtot * r / (np.linalg.norm(r)) ** 3])
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
# define Specific 2BodyRK integration
##########################################################
def I2BRK(r_arg, v_arg, M, G, tmax):
    # transform s-coordinates to relative coordinate
    r = r_arg[0, :] - r_arg[1, :]
    v = v_arg[0, :] - v_arg[1, :]
    # solve
    y0 = (np.array([r.flatten(), v.flatten()])).flatten()
    tspan = (0, tmax)
    t, y = ode45(lambda t, y: ODE2B(y, M, G), tspan, y0)

    # transform back to s-coordinates
    Mtot = sum(M)
    N = np.shape(t)[0]
    r1 = np.zeros((N, 3))
    r2 = np.zeros((N, 3))

    for j in range(N):
        r1[j, :] = y[j, :3] * M[1] / Mtot
        r2[j, :] = -y[j, :3] * M[0] / Mtot
    return [t, r1, r2]


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
        yi = y[i, :]
        sol = fsolve(
            lambda q: yi + dt * NBodyODE((yi + q) / 2.0, M, G, N) - q,
            y[i + 1, :],
        )
        y[i + 1, :] = sol
    if flag:
        return (t, y)
    else:
        return [t, y[-1][0]]
