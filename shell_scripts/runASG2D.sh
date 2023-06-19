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

##############
# Parameters #
##############

nPart=$1
seed=$2

rotD=$3

Rp=$4

initMode='L'
# can be:
#    'R' random, 
#    'S' restart from previous simulation,
#    'L' hexagonal lattice

couplingMode=$5
# can be:
#    'C' constant, 
#    'T' for two populations, 
#    'G' for Gaussian distribution, 
#    'F' for normally distributed ferromagnetic, 
#    'A' for normally distributed antiferromagnetic
# K0=$6

# KAA=10.0
# KAB=0.0
# KBB=10.0

KAVG=$6
STDK=$7

dT=1.e-4
DT=0.01
DTex=1.0
eqT=0
simulT=$8

savePos=1
saveForce=0
saveCoupling=0

# Cluster
if [ "${couplingMode}" == "C" ]; then
    run_dir=$HOME/2D_ActiveSpinGlass_lattice/simulation_data/Constant/N${nPart}/K${K0}/Rp${Rp}/rotD${rotD}/s${seed}
elif [ "${couplingMode}" == "T" ]; then
    run_dir=$HOME/2D_ActiveSpinGlass_lattice/simulation_data/TwoPopulations/N${nPart}/N${nPart}/K${KAA}/Rp${Rp}/rotD${rotD}/s${seed}
elif [ "${couplingMode}" == "G" ]; then
    run_dir=$HOME/2D_ActiveSpinGlass_lattice/simulation_data/Gaussian/N${nPart}/K${KAVG}_${STDK}/Rp${Rp}/rotD${rotD}/s${seed}
elif [ "${couplingMode}" == "F" ]; then
    run_dir=$HOME/2D_ActiveSpinGlass_lattice/simulation_data/Gaussian/N${nPart}/K${KAVG}_${STDK}/Rp${Rp}/rotD${rotD}/s${seed}
elif [ "${couplingMode}" == "A" ]; then
    run_dir=$HOME/2D_ActiveSpinGlass_lattice/simulation_data/Gaussian/N${nPart}/K${KAVG}_${STDK}/Rp${Rp}/rotD${rotD}/s${seed}
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

if [ -e "stats" ]; then
    rm 'stats'
fi

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
