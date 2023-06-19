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
# bin_dir=$HOME/Code/2D_ActiveSpinGlass_EL/bin

##############
# Parameters #
##############

nPart=$1
phi=$2
noise=$3

seed=$4

vp=$5

Rp=1.0
xTy=$6

initMode='R'
# can be:
#    'R' random, 
#    'S' restart from previous simulation

couplingMode=$7
# can be:
#    'C' constant, 
#    'T' for two populations, 
#    'G' for Gaussian distribution, 
#    'F' for normally distributed ferromagnetic, 
#    'A' for normally distributed antiferromagnetic

K0=$8

# KAA=10.0
# KAB=0.0
# KBB=10.0

# KAVG=$8
# STDK=$9

couplingModeNew=$9
# can be:
#    'C' constant, 
#    'G' for Gaussian distribution

# K0_new=1.0

KAVG_new=${10}
STDK_new=${11}

dT=${12}
DT=${13}
DTex=${14}
eqT=${15}
switchT=${16}
simulT=${17}

savePos=1
saveForce=0
saveCoupling=0 # Need to save couplings to be able to restart sim later for e.g. mode 'G'
saveCouplingNew=0

intMethod=${18}

# Local
if [ "${couplingMode}" == "C" ]; then
    if [ "${couplingModeNew}" == "C" ]; then
        run_dir=$HOME/Vicsek_2D/simulation_data/Switch_CC/N${nPart}/phi${phi}_n${noise}/K${K0}/Knew${K0_new}/xTy${xTy}/s${seed}
    elif [ "${couplingModeNew}" == "G" ]; then
        run_dir=$HOME/Vicsek_2D/simulation_data/Switch_CG/N${nPart}/phi${phi}_n${noise}/K${K0}/Knew${KAVG_new}_${STDK_new}/xTy${xTy}/s${seed}
    fi
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

echo "Starting Vicsek 2D run..."

cd $run_dir

# if [ -e "stats" ]; then
#     rm 'stats'

if [ ${initMode} == "S" ]; then # Only overwrite initMode and simulT in inpar if restarting from previous simulation
    sed -i '' "7s/.*/${initMode}/" 'inpar' # extra '' required on MacOS for sed (remove on Linux)
    if [ "${couplingMode}" == "C" ]; then
        sed -i '' "14s/.*/${simulT}/" 'inpar'
    elif [ "${couplingMode}" == "T" ]; then
        sed -i '' "16s/.*/${simulT}/" 'inpar'
    elif [ "${couplingMode}" == "G" ]; then
        sed -i '' "15s/.*/${simulT}/" 'inpar'
    elif [ "${couplingMode}" == "F" ]; then
        sed -i '' "15s/.*/${simulT}/" 'inpar'
    elif [ "${couplingMode}" == "A" ]; then
        sed -i '' "15s/.*/${simulT}/" 'inpar'
    fi

else
    if [ -e "inpar" ]; then
        rm 'inpar'
    fi
    touch 'inpar'

    echo ${nPart} > 'inpar'
    echo ${phi} >> 'inpar'
    echo ${seed} >> 'inpar'

    echo ${noise} >> 'inpar'
    echo ${vp} >> 'inpar'
    echo ${Rp} >> 'inpar'
    echo ${xTy} >> 'inpar'

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
    echo ${couplingModeNew} >> 'inpar'
    if [ "${couplingModeNew}" == "C" ]; then
        echo ${K0_new} >> 'inpar'
    elif [ "${couplingModeNew}" == "G" ]; then
        echo ${KAVG_new} >> 'inpar'
        echo ${STDK_new} >> 'inpar'
    fi

    echo ${dT} >> 'inpar'
    echo ${DT} >> 'inpar'
    echo ${DTex} >> 'inpar'
    echo ${eqT} >> 'inpar'
    echo ${switchT} >> 'inpar'
    echo ${simulT} >> 'inpar'

    echo ${savePos} >> 'inpar'
    echo ${saveForce} >> 'inpar'
    echo ${saveCoupling} >> 'inpar'
    echo ${saveCouplingNew} >> 'inpar'

    echo ${intMethod} >> 'inpar'
fi

time ${bin_dir}/vicsek_2D_switch inpar

echo "2D Vicsek run done."
