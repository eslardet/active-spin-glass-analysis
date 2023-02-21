#!/bin/bash

#  runASG2D.sh
#  ===========
#  Created by Thibault Bertrand on 2022-04-20
#  Copyright 2022 Imperial College London. All rights reserved.
#  Last modified on 2022-04-20

###############
# Directories #
###############

bin_dir=$HOME/Code/2D_ActiveSpinGlass_EL/bin
# matlab_dir=$bd_dir/codes/matlab

##############
# Parameters #
##############

nPart=1000
phi=0.6
seed=1

gx=1.0
Pe=120.0
Rr=1.0
Rp=2.0
xTy=1.0

initMode='R'
# can be:
#    'R' random, 
#    'S' restart from previous simulation

couplingMode='C'
# can be:
#    'C' constant, 
#    'T' for two populations, 
#    'G' for Gaussian distribution, 
#    'F' for normally distributed ferromagnetic, 
#    'A' for normally distributed antiferromagnetic

K0=0.0

# KAA=10.0
# KAB=0.0
# KBB=10.0

# KAVG=1.0
# STDK=1.0

dT=2.e-5
DT=0.1
DTex=1.0
eqT=0
simulT=1.0

savePos=1
saveForce=0
saveCoupling=0 # Need to save couplings to be able to restart sim later for e.g. mode 'G'

intMethod='E'

potMode='W'
# can be:
#    'W' WCA potential,
#    'H' Harmonic potential
#    'C' Continuous potential (repulsive part of WCA) with set cutoff

# Local
if [ "${couplingMode}" == "C" ]; then
    run_dir=$HOME/Code/2D_ActiveSpinGlass_EL/simulation_data_mips/Constant/N${nPart}/phi${phi}_Pe${Pe}/K${K0}/s${seed}
elif [ "${couplingMode}" == "T" ]; then
    run_dir=$HOME/Code/2D_ActiveSpinGlass_EL/simulation_data_mips/TwoPopulations/N${nPart}/phi${phi}_Pe${Pe}/K${KAA}/s${seed}
elif [ "${couplingMode}" == "G" ]; then
    run_dir=$HOME/Code/2D_ActiveSpinGlass_EL/simulation_data_mips/Gaussian/N${nPart}/phi${phi}_Pe${Pe}/K${KAVG}_${STDK}/s${seed}
elif [ "${couplingMode}" == "F" ]; then
    run_dir=$HOME/Code/2D_ActiveSpinGlass_EL/simulation_data_mips/Ferromagnetic/N${nPart}/phi${phi}_Pe${Pe}/K${KAVG}_${STDK}/s${seed}
elif [ "${couplingMode}" == "A" ]; then
    run_dir=$HOME/Code/2D_ActiveSpinGlass_EL/simulation_data_mips/Antiferromagnetic/N${nPart}/phi${phi}_Pe${Pe}/K${KAVG}_${STDK}/s${seed}
fi


###################################
# Create directories if necessary #
###################################
if [ ! -d "$run_dir" ]; then
    mkdir -p $run_dir
    echo "Creating directory : $run_dir"
fi

############################
# Run 2D Active Spin Glass #
############################

echo "Starting 2D Active Spin Glass run..."

cd $run_dir

if [ ${initMode} == "S" ]; then # Only overwrite initMode and simulT in inpar if restarting from previous simulation
    sed -i '' "9s/.*/${initMode}/" 'inpar' # extra '' required on MacOS for sed (remove on Linux)
    sed -i '' "14s/.*/${simulT}/" 'inpar'

else
    if [ -e "inpar" ]; then
        rm 'inpar'
    fi
    touch 'inpar'

    echo ${nPart} > 'inpar'
    echo ${phi} >> 'inpar'
    echo ${seed} >> 'inpar'

    echo ${gx} >> 'inpar'
    echo ${Pe} >> 'inpar'
    echo ${Rr} >> 'inpar'
    echo ${xTy} >> 'inpar'

    echo ${initMode} >> 'inpar'

    echo ${dT} >> 'inpar'
    echo ${DT} >> 'inpar'
    echo ${DTex} >> 'inpar'
    echo ${eqT} >> 'inpar'
    echo ${simulT} >> 'inpar'

    echo ${savePos} >> 'inpar'
    echo ${saveForce} >> 'inpar'

    echo ${intMethod} >> 'inpar'

    echo ${potMode} >> 'inpar'
fi

time ${bin_dir}/activeSpinGlass_2D_mips inpar

echo "2D Active Spin Glass run done."