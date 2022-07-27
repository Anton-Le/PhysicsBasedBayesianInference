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

Installation manual for MVAPICH2. 

#### Step 1: Configuration
Requires automake >= 1.15, libtool >= 2.4.5
Requires YACC/Bison.

Set the path where MPI should be installed to:
`export MPI_ROOT=<path to directory>`

**Note**: System-wide installation (default, without a specific `--prefix`) are discouraged as they will likely interfere with whatever
the distribution's package manager has installed.

The following should work in the directory where the MVAPICH source files were extracted to:

```
./configure --prefix=$MPI_ROOT --enable-cxx --enable-fortran=yes --enable-fast=all,O3 --enable-error-checking=runtime --enable-error-messages=generic --enable-timing=runtime --enable-g=none --enable-threads=runtime --enable-dependency-tracking --disable-rdma-cm
```

To disable FORTRAN support _and_ use an almost default installation use:
```
./configure --prefix=$MPI_ROOT --enable-cxx --enable-fast=all,O3 --enable-threads=runtime --enable-dependency-tracking --disable-rdma-cm --disable-fortran
```
If you recieve an error relating to infinityband, check you have libibverbs installed.

#### Step 2: Make, check and install

The common trifecta of building from source:

`make -j 10`
`make check`

Results should be along the lines of:
```
============================================================================
Testsuite summary for MPL 0.1
============================================================================
# TOTAL: 1
# PASS:  1
# SKIP:  0
# XFAIL: 0
# FAIL:  0
# XPASS: 0
# ERROR: 0
============================================================================
```

`make installcheck`

`make install`

#### Step 3: system configuration

First we must make the library known system-wide on-demand. To this end crate
a BASH script with the following contents:

```
#!/bin/bash

export MPI_ROOT=<your chosen path>
export PATH=$MPI_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib64:$LD_LIBRARY_PATH
export CPATH=$MPI_ROOT/include:$CPATH
export MANPATH=$MPI_ROOT/share/man:$MANPATH

## auxiliary environmental variables
export MPICC=mpicc
export MPICXX=mpicxx
export MPIF77=mpifort
export MPI_COMPILER=mpich3

export MPI_HOME=$MPI_ROOT
```

Store the script as `MVAPICHmodule.sh` and source it to set this 
as default MPI via:
`source MVAPICHmodule.sh`

Test via:
`which mpicxx`

This should print `$PATH/mpicxx`, e.g.:
`/programs/libraries/bin/MVAPICH/2.3.6/bin/mpicxx`

and `mpicxx --version` should list your system's default compiler as the first line,
i.e. `g++ (SUSE Linux) 7.5.0`.

#### Step 4: runtime configuration

When running manually via `mpiexec -n .. <program> <options>` MPI will require a hostfile.
On clusters this is generally supplied by the scheduler (e.g. SLURM).
On a workstation (or a small office cluster) this can be a simple text file called, e.g., `local.node`
and containinig only one line: `127.0.0.1:6`

Here the first part is the IP address  (`127.0.0.1` for the local host) and the part after
the colon denotes how many MPI processes (slots) can be started on the node. The number of slots is arbitrary and
as such can be set much higher than the actuall processor count of the system. This carries with it
the potential for oversubscription of resources and slowdown due to resource contention.
I hence suggest setting the number of slots to the number of **physical** CPU cores, although
using the number of **logical** CPU cores (e.g., threads) should work, too.

To run an MPI program will then require the following command:
`mpiexec -n <number of processes> -h /path/to/local.node <program name> <program options>`

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
