import torch
from torch.distributed import barrier
import os

from cmdline import *
from atlas import *

avgfile = f'{prefix}oasisavg_{suffix}.pth'
Iavg = torch.load(avgfile, map_location=loc)

reg_weightA=1e1
reg_weightT=1e1

convaffinefile = f'{prefix}convaffine_{suffix}.pth'
if not os.path.isfile(convaffinefile): # compute affine atlas
    print("Conventional affine atlas building")
    As = torch.zeros((len(oasis_ds),3,3), dtype=torch.float32).to(loc)
    Ts = torch.zeros((len(oasis_ds),3), dtype=torch.float32).to(loc)
    res = affine_atlas(
        dataset=oasis_ds,
        I=Iavg, As=As, Ts=Ts,
        affine_steps=1,
        num_epochs=250,
        learning_rate_A=1e-3,
        learning_rate_T=1e-2,
        learning_rate_I=2e4,
        reg_weightA=reg_weightA,
        reg_weightT=reg_weightT,
        batch_size=8,
        gpu=gpu,
        world_size=args.world_size,
        rank=rank)
    # save result
    if rank == 0: torch.save(res, convaffinefile)
barrier()

# On the test set, use same atlas-building code but with zero learning rate for
# the image
convaffinetestfile = f'{prefix}convaffine_test_{suffix}.pth'
if not os.path.isfile(convaffinetestfile): # conventional lddmm atlas
    I_affine, _, _ = torch.load(convaffinefile, map_location='cpu')
    print("Conventional Affine Test")
    As = torch.zeros((len(oasis_test_ds),3,3), dtype=torch.float32).to('cpu')
    Ts = torch.zeros((len(oasis_test_ds),3), dtype=torch.float32).to('cpu')
    res = affine_atlas(
        dataset=oasis_test_ds,
        I=I_affine, As=As, Ts=Ts,
        affine_steps=250,
        num_epochs=1,
        learning_rate_A=1e-3,
        learning_rate_T=1e-2,
        learning_rate_I=0e5,
        reg_weightA=reg_weightA,
        reg_weightT=reg_weightT,
        batch_size=8,
        gpu=gpu,
        world_size=args.world_size,
        rank=rank)
    if rank == 0: torch.save(res, convaffinetestfile)
    del res
