Linear model with provision of initial data.

This version of the model is a non-vectorized distribution of the
data. with a section for transformed data, since we may not assume
that measurements start at t_0 = 0.


And it utilizes an initial guess for the parameters.


## Generated data

Data generated with Python using the foling parameters:

g = -9.80665 # [m/s^2]
v0 = 78 # [m/s]
z0 = 100 # [m]

dt = 0.01
N = 200
sigmaObs = 3 # [m/s]
W = np.random.rand(0, sigmaObs, size=N)

z = z0 + v0 * t + 0.5 * g * t**2 + W;
