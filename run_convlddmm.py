import torch
from torch.distributed import barrier
import os

from cmdline import *
from atlas import *

oasis_ds_std = OASISDataset(crop=None,
                        h5path=f'{prefix}deepaffinestd_{suffix}.h5',
                        pooling=None,
                        one_scan_per_subject=False)
oasis_ds_test_std = OASISDataset(crop=None,
                        h5path=f'{prefix}deepaffinestd_test_{suffix}.h5',
                        pooling=None,
                        one_scan_per_subject=False)

deepaffinefile = f'{prefix}deepaffine_{suffix}.pth'
I_deepaffine, affine_net, epoch_losses_deepaffine, full_losses_deepaffine, \
            iter_losses_deepaffine, test_losses_deepaffine \
        = torch.load(deepaffinefile, map_location=loc)
I_deepaffine = I_deepaffine.to(loc)

fluid_params = [.1,0,.01]
if rank == 0: torch.save(fluid_params, f'{prefix}fluidparams_{suffix}.pth')
convlddmmfile = f'{prefix}convlddmm_{suffix}.pth'
if not os.path.isfile(convlddmmfile): # conventional lddmm atlas
    print("Conventional LDDMM atlas building")
    res = lddmm_atlas(dataset=oasis_ds_std,
                                          I0=I_deepaffine.clone().to('cuda'),
                                          fluid_params=fluid_params,
                                          learning_rate_pose=1e-3,
                                          learning_rate_image=5e4,
                                          reg_weight=1e2,
                                          momentum_preconditioning=False,
                                          batch_size=30,
                                          num_epochs=500,
                                          gpu=gpu,
                                          world_size=args.world_size,
                                          rank=rank)
    if rank == 0: torch.save(res, convlddmmfile)
    del res
barrier()
Ilddmm, _, _, _ = torch.load(convlddmmfile, map_location='cpu')
Ilddmm = Ilddmm.to(loc)

# On the test set, use same atlas-building code but with zero learning rate for
# the image
convlddmmtestfile = f'{prefix}convlddmm_test_{suffix}.pth'
if not os.path.isfile(convlddmmtestfile): # conventional lddmm atlas
    print("Conventional LDDMM Test")
    res = lddmm_atlas(dataset=oasis_ds_test_std,
                                          I0=Ilddmm,
                                          fluid_params=fluid_params,
                                          learning_rate_pose=1e-3,
                                          learning_rate_image=0e4,
                                          momentum_preconditioning=False,
                                          reg_weight=1e2,
                                          batch_size=30,
                                          num_epochs=1,
                                          lddmm_steps=500,
                                          gpu=gpu,
                                          world_size=args.world_size,
                                          rank=rank)
    if rank == 0: torch.save(res, convlddmmtestfile)
    del res
#Ilddmm, mom_lddmm, epoch_losses, iter_losses = torch.load(convlddmmtestfile,
        #map_location=loc)
