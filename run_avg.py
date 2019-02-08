import torch
import os

from cmdline import *
from oasis_data import *

# compute avg image
batch_size=12
avgfile = f'{prefix}oasisavg_{suffix}.pth'
if not os.path.isfile(avgfile):
    if rank == 0:
        print("Voxel averaging")
        Iavg = batch_average(DataLoader(oasis_ds, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False), dim=0)
    if rank == 0: torch.save(Iavg, avgfile)
