# Set-up of the development environment

## Preface

In the following a way to set up a development environment on Linux is described.
It is _assumed_ that a system-wide installation of Python 3 is available. On a cluster
this is not necessarily the case and a python module should be loaded with, e.g., `module load python`.

The following steps utilise Python 3.8.2.
For starters MPI requirements can be ignored.

## Step 1 - Virtual environment
Create a folder for the virtual environment:
`mkdir NumPyroEnv`

Initialize the environment:
`python3 -m venv /programs/extension/tmp/NumPyroEnv`

This will create a virtual Python installation in the selected folder that will utilise
the system-wide Python binary while compartmentizing the user-installed modules to the
chosen directory.

Activate the virtual environment:

`source /programs/extension/tmp/NumPyroEnv/bin/activate`

Check the installed libraries
`pip3 list`

This should produce something to the tune of:
```
Package    Version
---------- -------
pip        19.2.3 
setuptools 41.2.0 
WARNING: You are using pip version 19.2.3, however version 22.1.2 is available.
You should consider upgrading via the 'pip install --upgrade pip' command.
```

It is advisable to update PIP:
`pip3 install --upgrade pip`

## Step 2 - Dependencies

Common libraries that can be installed without a second thought are
  - NumPy
  - SciPy
  - Matplotlib
  - Cython
`pip3 install numpy scipy matplotlib cython`
Will produce the output:
```
Installing collected packages: six, pyparsing, pillow, numpy, kiwisolver, fonttools, cycler, scipy, python-dateutil, packaging, matplotlib
Successfully installed cycler-0.11.0 fonttools-4.34.4 kiwisolver-1.4.3 matplotlib-3.5.2 numpy-1.23.1 packaging-21.3 pillow-9.2.0 pyparsing-3.0.9 python-dateutil-2.8.2 scipy-1.8.1 six-1.16.0
```


### JAX

NumPyro requires JAX to be installed. Since the ultimate goal of this project is to run on GPUs on multiple compute nodes (in a cluster)
the library should be installed with GPU support. This requires an installation of CuDNN.

For starters the installation can be performed for CPU only:

`pip install --upgrade "jax[cpu]"`

The output should be something along the lines of
```
Installing collected packages: flatbuffers, zipp, typing_extensions, opt_einsum, etils, absl-py, jaxlib, importlib_resources, jax
  Running setup.py install for jax ... done
Successfully installed absl-py-1.1.0 etils-0.6.0 flatbuffers-2.0 importlib_resources-5.8.0 jax-0.3.14 jaxlib-0.3.14 opt_einsum-3.3.0 typing_extensions-4.3.0 zipp-3.8.0
```

### NumPyro

In case JAX has been installed for the CPU only NumPyro can be installed with CPU support:

`pip install numpyro[cpu]`

### MPI

tbd. (assuming a system-wide installation exists)

### Python MPI

The general purpose library for MPI communications in Python is `mpi4py`.
For JAX a specific interface to MPI is available under the name `mpi4jax`.

**Assuming** MPI is installed on the system the libraries can be installed via:

`pip3 install mpi4py`
`pip3 install mpi4jax --no-build-isolation`

**NOTE**: The previous command may fail with the following error:
`error: invalid command 'bdist_wheel'`

In this case `wheel` needs to be installed via
`pip3 install wheel`
and the installation of `mpi4jax` re-run.