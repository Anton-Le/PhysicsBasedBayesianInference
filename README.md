# PhysicsBasedBayesianInference
Implementation of ensemble-based Hamiltonian Monte Carlo for multiple architectures.

## Purpose

This repository contains simple implementations of Hamiltonian Monte Carlo and
its ensemble-based extensions in Python. The implementations shall include physical constants
as they are intended to enable the application of physical intuition to the problem
of determining the parameters of stochastic models using Bayesian inference.

The code utilises (NumPyro)[https://num.pyro.ai/en/stable/] to allow a user to define a
probabilistic model in the `STAN` probabilistic programming language ( a derivative of C++ )
that will then be fit using the implemented methods. The intent is to widen the range
of architectures that will be usable by an existing model base.

Note that some of the methods implemented here are already present in the above framework,
albeit after undergoing 'mathematical mutilation'.

# Repository structure

This repository consists of 3 main branches
1. `main` - Containing this readme as well as miscellaneous information. The "release" branch.
2. `dev` - The development branch.

Any other branches may be created but will be considered transients subject to removal
at a later date.

The `dev` branch is to be used for code development and shall contain code approved by at least one
reviewer. The `main` branch will consist of code that represents major milestones of development
and has passed _review and testing_.

This repository contains the following directories:
  - `references`: directory with reference papers, notes and a work plan.
  - `src`: the main source directory
  - `samples`: contains sample code from previous projects that can be taken as reference

# Conventions 

Formatting guidelines and code contribution procedures are to be found in RulesAndProcedures.md