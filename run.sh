#!/bin/bash -l

set -ex

source /ccs/proj/med107/software/pytorch-1.0-p3/source_to_run_pytorch1.0-p3

cd /gpfs/alpine/proj-shared/med107/4jh/midl/diffeomorphic_autoencoders

#env

MASTER_ADDR=$1
NFOLDS=$2
FOLD=$3
SCRIPT=$4

GPUS_PER_NODE=6

if [ "$OMPI_COMM_WORLD_SIZE" == "1" ]
then # single node

    /ccs/home/4jh/summitenvs/lagomorph/bin/python -m torch.distributed.launch \
        --nproc_per_node=$GPUS_PER_NODE \
            $SCRIPT --num_folds $NFOLDS --fold $FOLD \
                --world_size=$GPUS_PER_NODE \
                --nprocs_per_node=$GPUS_PER_NODE

else # multi-node

    FINAL_WORLD_SIZE=$((OMPI_COMM_WORLD_SIZE * GPUS_PER_NODE))
    PORT=1234
    NODE_RANK=$OMPI_COMM_WORLD_RANK
    /ccs/home/4jh/summitenvs/lagomorph/bin/python -m torch.distributed.launch \
        --nproc_per_node=$GPUS_PER_NODE \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR --master_port=$PORT \
            $SCRIPT --num_folds $NFOLDS --fold $FOLD \
                --world_size=$FINAL_WORLD_SIZE \
                --nprocs_per_node=$GPUS_PER_NODE \
                --node_rank=$NODE_RANK
fi

