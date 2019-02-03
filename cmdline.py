import torch
import random
import lagomorph
import numpy as np
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_folds','-k', default=5, type=int, help="Number of cross-validation folds")
parser.add_argument('--fold','-f', default=0, type=int, help="Which fold to process (zero-indexed)")
parser.add_argument('--node_rank','-r', default=0, type=int, help="Rank of main process on this node")
parser.add_argument('--local_rank','-g', default=None, type=int, help="Local rank, i.e. which GPU to use")
parser.add_argument('--world_size','-w', default=1, type=int, help="Total number of processes launched")
parser.add_argument('--nprocs_per_node','-n', default=1, type=int, help="Total number of processes launched on each node")
args = parser.parse_args()


if args.local_rank is not None:
    gpu = args.local_rank
    torch.cuda.set_device(args.local_rank)
else:
    gpu = 0
# compute overall rank
rank = args.node_rank*args.nprocs_per_node + args.local_rank
if args.world_size is not 1:
    torch.distributed.init_process_group(backend='nccl',
            world_size=args.world_size,
            init_method='env://')
loc = f'cuda:{gpu}'

print(f"World size: {args.world_size} Local rank: {args.local_rank} Gpu: {gpu} Node Rank: {args.node_rank} PPN: {args.nprocs_per_node} Global Rank: {rank}")

# try to be as reproducible as possible. Due to atomics used in backpropagation,
# even this is not enough for bit-for-bit reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

from oasis_data import *
from atlas import *
from deepatlas import *

num_folds = args.num_folds
fold = args.fold
one_scan_per_subject = False
s = 2
prefix = f'output/oasis3_downscale{s}_cv/{num_folds}/{fold}/'
os.makedirs(prefix, exist_ok=True)
datadir = f'data/oasis3_downscale{s}_cv/{num_folds}'
crop //= s
def get_dataset(split):
    h5path = f'{datadir}/{split}{fold}.h5'
    return OASISDataset(crop=crop,
                        h5path=h5path,
                        pooling=ds_pooling,
                        one_scan_per_subject=one_scan_per_subject)
oasis_ds = get_dataset('train')
oasis_test_ds = get_dataset('test')
l = len(oasis_ds)
sz = oasis_ds[0][1].shape[2]
suffix = f'crop{docrop}_oneper{one_scan_per_subject}_{sz}_{l}'

