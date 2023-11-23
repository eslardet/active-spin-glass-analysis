# Active Spin Glass

## Overview
Agent based modelling of aligning self-propelled particles with random couplings. Simulations are run in C++ using source code in `src`. 
This must first be compiled using `make -f Makefile` in the relevant folder, then run using one of the `runV.sh` scripts. 
Additionally, code can be run from a HPC using a `bashV.pbs` script. This is especially useful for job parallisation for different seed realizations or arrays of parameters.

Once simulations have been run and text files have been generated, these data can be analysed and plotted using the various analysis functions in the `analysis` folder.
