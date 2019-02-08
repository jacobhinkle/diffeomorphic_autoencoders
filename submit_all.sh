#!/bin/bash

source /ccs/proj/med107/software/pytorch-1.0-p3/source_to_run_pytorch1.0-p3 
export PATH=$HOME/summitenvs/lagomorph/bin:$PATH

NFOLDS=${1:-5}
FOLD=${2:-0}

JOB_PREFIX=diffae_${NFOLDS}_${FOLD}

job() {
    NNODES=$1
    shift
    WALLTIME=$1
    shift
    JOBNAME=$1
    shift
    SCRIPT=$1
    shift
    JOBFILE=.${JOBNAME}.lsf
    # create job script from template
    CFGJSON="{\"script\":\"${SCRIPT}\",\"nnodes\":${NNODES},\"nfolds\":${NFOLDS},\"fold\":${FOLD},\"jobname\":\"${JOBNAME}\"}"
    pystache run_script.lsf.mustache $CFGJSON > $JOBFILE
    bsub -J $JOBNAME -W $WALLTIME $* $JOBFILE
    rm $JOBFILE
}

# submit jobs with proper dependencies

#job 1 0:10 ${JOB_PREFIX}_avg run_avg.py
#job 1 2:00 ${JOB_PREFIX}_convaffine run_convaffine.py \
#    -w ${JOB_PREFIX}_avg
#job 1 2:00 ${JOB_PREFIX}_deepaffine run_deepaffine.py \
#    -w ${JOB_PREFIX}_avg
#job 16 2:00 ${JOB_PREFIX}_convlddmm run_convlddmm.py \
    #-w ${JOB_PREFIX}_deepaffine
#job 16 2:00 ${JOB_PREFIX}_deeplddmm run_deeplddmm.py \
#    -w ${JOB_PREFIX}_deepaffine

job 1 0:30 ${JOB_PREFIX}_avg run_avg.py
#job 1 1:00 ${JOB_PREFIX}_convaffine run_convaffine.py \
#    -w ${JOB_PREFIX}_avg
#job 1 1:00 ${JOB_PREFIX}_deepaffine run_deepaffine.py \
    #-w ${JOB_PREFIX}_avg
#job 1 1:00 ${JOB_PREFIX}_convlddmm run_convlddmm.py \
#    -w ${JOB_PREFIX}_deepaffine
#job 1 1:00 ${JOB_PREFIX}_deeplddmm run_deeplddmm.py \
#    -w ${JOB_PREFIX}_deepaffine
