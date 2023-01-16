#!/bin/bash

#  runASG2D.sh
#  ===========
#  Created by Thibault Bertrand on 2022-04-20
#  Copyright 2022 Imperial College London. All rights reserved.
#  Last modified on 2022-04-20

###############
# Directories #
###############

bin_dir=$HOME/bin
# matlab_dir=$bd_dir/codes/matlab

##############
# Parameters #
##############

nPart=$1
phi=0.2
seed=$2

gx=1.0
Pe=$3
Rr=1.0
Rp=5.0
xTy=$4

initMode='R'
potMode='H'
# can be:
#    'W' WCA potential,
#    'H' Harmonic potential
couplingMode='C'
# can be:
#    'C' constant, 
#    'T' for two populations, 
#    'G' for Gaussian distribution, 
#    'F' for normally distributed ferromagnetic, 
#    'A' for normally distributed antiferromagnetic
K0=$5

# KAA=10.0
# KAB=0.0
# KBB=10.0

#KAVG=$3
#STDK=$4

dT=2.e-5
DT=0.01
eqT=0
simulT=20

savePos=1
saveForce=0
saveCoupling=0

# Cluster
if [ "${couplingMode}" == "C" ]; then
    run_dir=$HOME/2D_ActiveSpinGlass_EL/simulation_data/Constant/N${nPart}/phi${phi}_Pe${Pe}/K${K0}/s${seed}
elif [ "${couplingMode}" == "T" ]; then
    run_dir=$HOME/2D_ActiveSpinGlass_EL/simulation_data/TwoPopulations/N${nPart}/phi${phi}_Pe${Pe}/K${KAA}/s${seed}
elif [ "${couplingMode}" == "G" ]; then
    run_dir=$HOME/2D_ActiveSpinGlass_EL/simulation_data/Gaussian/N${nPart}/phi${phi}_Pe${Pe}/K${KAVG}_${STDK}/s${seed}
elif [ "${couplingMode}" == "F" ]; then
    run_dir=$HOME/2D_ActiveSpinGlass_EL/simulation_data/Ferromagnetic/N${nPart}/phi${phi}_Pe${Pe}/K${KAVG}_${STDK}/s${seed}
elif [ "${couplingMode}" == "A" ]; then
    run_dir=$HOME/2D_ActiveSpinGlass_EL/simulation_data/Antiferromagnetic/N${nPart}/phi${phi}_Pe${Pe}/K${KAVG}_${STDK}/s${seed}
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
echo ${Rp} >> 'inpar'
echo ${xTy} >> 'inpar'

echo ${initMode} >> 'inpar'

echo ${potMode} >> 'inpar'

echo ${couplingMode} >> 'inpar'
if [ "${couplingMode}" == "C" ]; then
    echo ${K0} >> 'inpar'
elif [ "${couplingMode}" == "T" ]; then
    echo ${KAA} >> 'inpar'
    echo ${KAB} >> 'inpar'
    echo ${KBB} >> 'inpar'
elif [ "${couplingMode}" == "G" ]; then
    echo ${KAVG} >> 'inpar'
    echo ${STDK} >> 'inpar'
elif [ "${couplingMode}" == "F" ]; then
    echo ${KAVG} >> 'inpar'
    echo ${STDK} >> 'inpar'
elif [ "${couplingMode}" == "A" ]; then
    echo ${KAVG} >> 'inpar'
    echo ${STDK} >> 'inpar'
fi

echo ${dT} >> 'inpar'
echo ${DT} >> 'inpar'
echo ${eqT} >> 'inpar'
echo ${simulT} >> 'inpar'

echo ${savePos} >> 'inpar'
echo ${saveForce} >> 'inpar'
echo ${saveCoupling} >> 'inpar'

time ${bin_dir}/activeSpinGlass_2D_soft inpar

echo "2D Active Spin Glass run done."
