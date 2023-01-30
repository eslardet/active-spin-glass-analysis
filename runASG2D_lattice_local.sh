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
seed=1

rotD=1.0

Rp=2.0

initMode='L'
# can be:
#    'L' hexagonal lattice

couplingMode='C'
# can be:
#    'C' constant, 
#    'T' for two populations, 
#    'G' for Gaussian distribution, 
#    'F' for normally distributed ferromagnetic, 
#    'A' for normally distributed antiferromagnetic

K0=1.0

# KAA=10.0
# KAB=0.0
# KBB=10.0

# KAVG=1.0
# STDK=1.0

dT=2.e-5
DT=0.1
DTex=0.5
eqT=0
simulT=1.0

savePos=1
saveForce=0
saveCoupling=1 # Need to save couplings to be able to restart sim later for e.g. mode 'G'

# Local
if [ "${couplingMode}" == "C" ]; then
    run_dir=$HOME/Code/2D_ActiveSpinGlass_EL/simulation_data_lattice/Constant/N${nPart}/K${K0}/Rp${Rp}/s${seed}
elif [ "${couplingMode}" == "T" ]; then
    run_dir=$HOME/Code/2D_ActiveSpinGlass_EL/simulation_data_lattice/TwoPopulations/N${nPart}/K${KAA}/Rp${Rp}/s${seed}
elif [ "${couplingMode}" == "G" ]; then
    run_dir=$HOME/Code/2D_ActiveSpinGlass_EL/simulation_data_lattice/Gaussian/N${nPart}/K${KAVG}_${STDK}/Rp${Rp}/s${seed}
elif [ "${couplingMode}" == "F" ]; then
    run_dir=$HOME/Code/2D_ActiveSpinGlass_EL/simulation_data_lattice/Ferromagnetic/N${nPart}/K${KAVG}_${STDK}/Rp${Rp}/s${seed}
elif [ "${couplingMode}" == "A" ]; then
    run_dir=$HOME/Code/2D_ActiveSpinGlass_EL/simulation_data_lattice/Antiferromagnetic/N${nPart}/K${KAVG}_${STDK}/Rp${Rp}/s${seed}
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
echo ${seed} >> 'inpar'

echo ${rotD} >> 'inpar'

echo ${Rp} >> 'inpar'

echo ${initMode} >> 'inpar'

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
echo ${DTex} >> 'inpar'
echo ${eqT} >> 'inpar'
echo ${simulT} >> 'inpar'

echo ${savePos} >> 'inpar'
echo ${saveForce} >> 'inpar'
echo ${saveCoupling} >> 'inpar'

time ${bin_dir}/activeSpinGlass_2D_lattice inpar

echo "2D Active Spin Glass run done."
