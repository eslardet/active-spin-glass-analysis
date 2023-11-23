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
phi=$2
noise=$3

seed=$4

vp=$5

Rp=$6
xTy=$7

initMode=$8
# can be:
#    'R' random, 
#    'A' aligned,
#    'S' restart from previous simulation

couplingMode=$9
# can be:
#    'C' constant, 
#    'T' for two populations, 
#    'G' for Gaussian distribution, 
#    'F' for normally distributed ferromagnetic, 
#    'A' for normally distributed antiferromagnetic

# KAA=10.0
# KAB=0.0
# KBB=10.0

KAVG=${10}
STDK=${11}

dT=${12}
DT=${13}
DTex=${14}
eqT=${15}
simulT=${16}

savePos=${17}
saveInitPos=${18}
saveForce=0
saveCoupling=${19}

intMethod=${20}

if [ "${couplingMode}" == "C" ]; then
    run_dir=$HOME/ASG_2D/simulation_data/Constant/N${nPart}/phi${phi}_n${noise}/K${K0}/Rp${Rp}/xTy${xTy}/s${seed}
elif [ "${couplingMode}" == "T" ]; then
    run_dir=$HOME/ASG_2D/simulation_data/TwoPopulations/N${nPart}/phi${phi}_n${noise}/K${KAA}/Rp${Rp}/xTy${xTy}/s${seed}
elif [ "${couplingMode}" == "G" ]; then
    run_dir=$HOME/ASG_2D/simulation_data/Gaussian/N${nPart}/phi${phi}_n${noise}/K${KAVG}_${STDK}/Rp${Rp}/xTy${xTy}/s${seed}
elif [ "${couplingMode}" == "F" ]; then
    run_dir=$HOME/ASG_2D/simulation_data/Fraction/N${nPart}/phi${phi}_n${noise}/K${KAVG}_${STDK}/Rp${Rp}/xTy${xTy}/s${seed}
elif [ "${couplingMode}" == "A" ]; then
    run_dir=$HOME/ASG_2D/simulation_data/Asymmetric/N${nPart}/phi${phi}_n${noise}/K${KAVG}_${STDK}/Rp${Rp}/xTy${xTy}/s${seed}
fi

if [ "${initMode}" == 'A' ]; then
    run_dir+="_a"
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

echo "Starting ASG 2D run..."

cd $run_dir

if [ ${initMode} == "S" ]; then # Only overwrite initMode and simulT in inpar if restarting from previous simulation
    sed -i "8s/.*/${initMode}/" 'inpar' # extra '' required on MacOS for sed (remove on Linux)
    if [ "${couplingMode}" == "C" ]; then
        sed -i "12s/.*/${DT}/" 'inpar'
        sed -i "13s/.*/${DTex}/" 'inpar'
        sed -i "14s/.*/${eqT}/" 'inpar'
        sed -i "15s/.*/${simulT}/" 'inpar'
        sed -i "16s/.*/${savePos}/" 'inpar'
        sed -i "17s/.*/${saveInitPos}/" 'inpar'
        sed -i "19s/.*/${saveCoupling}/" 'inpar'
    elif [ "${couplingMode}" == "T" ]; then
        sed -i "14s/.*/${DT}/" 'inpar'
        sed -i "15s/.*/${DTex}/" 'inpar'
        sed -i "16s/.*/${eqT}/" 'inpar'
        sed -i "17s/.*/${simulT}/" 'inpar'
    elif [ "${couplingMode}" == "G" ]; then
        sed -i "13s/.*/${DT}/" 'inpar'
        sed -i "14s/.*/${DTex}/" 'inpar'
        sed -i "15s/.*/${eqT}/" 'inpar'
        sed -i "16s/.*/${simulT}/" 'inpar'
    elif [ "${couplingMode}" == "F" ]; then
        sed -i "13s/.*/${DT}/" 'inpar'
        sed -i "14s/.*/${DTex}/" 'inpar'
        sed -i "15s/.*/${eqT}/" 'inpar'
        sed -i "16s/.*/${simulT}/" 'inpar'
    elif [ "${couplingMode}" == "A" ]; then
        sed -i "13s/.*/${DT}/" 'inpar'
        sed -i "14s/.*/${DTex}/" 'inpar'
        sed -i "15s/.*/${eqT}/" 'inpar'
        sed -i "16s/.*/${simulT}/" 'inpar'
    fi

# else
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

echo ${dT} >> 'inpar'
echo ${DT} >> 'inpar'
echo ${DTex} >> 'inpar'
echo ${eqT} >> 'inpar'
echo ${simulT} >> 'inpar'

echo ${savePos} >> 'inpar'
echo ${saveInitPos} >> 'inpar'
echo ${saveForce} >> 'inpar'
echo ${saveCoupling} >> 'inpar'

echo ${intMethod} >> 'inpar'
# fi

time ${bin_dir}/asg_2D inpar

echo "2D ASG run done."
