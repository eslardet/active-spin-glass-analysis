# Active Spin Glass

## Overview
Agent based modelling of aligning self-propelled particles with random couplings. Simulations are run in C++ using source code in `src`. 
This must first be compiled using `make -f Makefile` in the relevant folder, then run using one of the `runV.sh` scripts. 
Additionally, code can be run from a HPC using a `bashV.pbs` script. This is especially useful for job parallelization for different seed realizations or arrays of parameters.

Once simulations have been run and text files have been generated, these data can be analyzed and plotted using the various analysis functions in the `analysis` folder.


## Model details
We model the microscopic dynamics using coupled overdamped Langevin equations with metric alignment interactions through a mean-sine force. These alignment interactions have coupling values $K_{ij}$, which can follow a number of distributions and can be changed with the `mode` parameter (e.g. constant, Gaussian).
The Langevin equations are integrated and solved numerically using either an Euler-Maruyama or Stochastic Runge-Kutta scheme.