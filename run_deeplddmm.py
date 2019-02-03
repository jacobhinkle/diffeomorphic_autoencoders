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
I_deepaffine, _, _, _, _, _ \
        = torch.load(deepaffinefile, map_location='cpu')

fluid_params = [.1,0,.01]
reg_weight = 1e2
lddmm_integration_steps = 5
batch_size = 2
deeplddmmfile = f'{prefix}deeplddmm_{suffix}.pth'
if not os.path.isfile(deeplddmmfile): # deep lddmm atlas
    print("Deep LDDMM atlas building")
    res = \
        deep_lddmm_atlas(oasis_ds_std,
                     I0=I_deepaffine,
                     #I=I_deeplddmm_ft.clone(),
                     num_epochs=500,
                     batch_size=batch_size,
                     closed_form_image=False,
                     image_update_freq=100,
                     reg_weight=reg_weight,
                     #momentum_net=copy.deepcopy(mom_net_ft),
                     momentum_preconditioning=False,
                     lddmm_integration_steps=lddmm_integration_steps,
                     learning_rate_pose=1e-6,
                     learning_rate_image=1e4,
                     fluid_params=fluid_params,
                     gpu=gpu,
                     world_size=args.world_size,
                     rank=rank)
    if rank == 0: torch.save(res, deeplddmmfile)
else:
    res = torch.load(deeplddmmfile, map_location=loc)
I_deeplddmm, mom_net, epoch_losses_deeplddmm, iter_losses_deeplddmm = res
barrier()

deeplddmmtestfile = f'{prefix}deeplddmm_test_{suffix}.pth'
if not os.path.isfile(deeplddmmtestfile): # deep lddmm atlas
    print("Deep LDDMM atlas building TEST")
    # manually do testing, just compute the regularization and image MSE here
    _, _, deeplddmm_test_loss, _ = \
        deep_lddmm_atlas(oasis_ds_test_std,
                     I0=I_deepaffine,
                     momentum_net = mom_net,
                     #I=I_deeplddmm_ft.clone(),
                     #num_epochs=500,
                     num_epochs=1,
                     batch_size=batch_size,
                     closed_form_image=False,
                     image_update_freq=100,
                     reg_weight=reg_weight,
                     #momentum_net=copy.deepcopy(mom_net_ft),
                     momentum_preconditioning=False,
                     lddmm_integration_steps=lddmm_integration_steps,
                     learning_rate_pose=0e-6,
                     learning_rate_image=0e4,
                     fluid_params=fluid_params,
                     gpu=gpu,
                     world_size=args.world_size,
                     rank=rank)
    if rank == 0: torch.save((deeplddmm_test_loss,), deeplddmmtestfile)
