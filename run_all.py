import torch
import numpy as np
import lagomorph
import os
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_folds','-k', default=5, type=int, help="Number of cross-validation folds")
parser.add_argument('--fold','-f', default=0, type=int, help="Which fold to process (zero-indexed)")
parser.add_argument('--local_rank','-g', default=None, type=int, help="Local rank, i.e. which GPU to use")
parser.add_argument('--rank', default=None, type=int, help="Global rank of this process")
parser.add_argument('--world_size','-w', default=1, type=int, help="Total number of processes launched")
args = parser.parse_args()

if args.local_rank is not None:
    torch.cuda.set_device(args.local_rank)
if args.rank is None:
    rank = args.local_rank
else:
    rank = args.rank
if args.world_size is not 1:
    torch.distributed.init_process_group(backend='nccl',
            world_size=args.world_size,
            init_method='env://')

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
h5path = f'data/oasis3_downscale{s}_cv/{num_folds}/train{fold}.h5'
crop //= s
oasis_ds = OASISDataset(crop=crop,
                        h5path=h5path,
                        pooling=ds_pooling,
                        one_scan_per_subject=one_scan_per_subject)
l = len(oasis_ds)
sz = oasis_ds[0][1].shape[2]
suffix = f'crop{docrop}_oneper{one_scan_per_subject}_{sz}_{l}'

# compute avg image
batch_size=12
avgfile = f'{prefix}oasisavg_{suffix}.pth'
if not os.path.isfile(avgfile):
    print("Voxel averaging")
    Iavg = batch_average(DataLoader(oasis_ds, batch_size=200, num_workers=8, pin_memory=True, shuffle=False), dim=0)
    torch.save(Iavg, avgfile)
Iavg = torch.load(avgfile)

convaffinefile = f'{prefix}convaffine_{suffix}.pth'
if not os.path.isfile(convaffinefile): # compute affine atlas
    print("Conventional affine atlas building")
    As = torch.zeros((len(oasis_ds),3,3), dtype=torch.float32).to('cuda')
    Ts = torch.zeros((len(oasis_ds),3), dtype=torch.float32).to('cuda')
    res = affine_atlas(
        dataset=oasis_ds,
        I=Iavg, As=As, Ts=Ts,
	affine_steps=1,
        num_epochs=1000,
        learning_rate_A=1e-4,
        learning_rate_T=1e-3,
        learning_rate_I=1e4,
        batch_size=50)
    # save result
    torch.save(res, convaffinefile)
Iaffine, epoch_losses_affine, iter_losses_affine = torch.load(convaffinefile)
Iaffine = Iaffine.to('cuda')

deepaffinefile = f'{prefix}deepaffine_{suffix}.pth'
if not os.path.isfile(deepaffinefile): # compute deep affine atlas
    print("Deep affine atlas building")
    res = \
	deep_affine_atlas(oasis_ds,
                      I=Iavg.clone(),
                      learning_rate_pose=1e-4,
                      learning_rate_image=1e4,
                      num_epochs=1000,
                      batch_size=50)
    torch.save(res, deepaffinefile)
I_deepaffine, affine_net, epoch_losses_deepaffine, full_losses_deepaffine, \
            iter_losses_deepaffine \
        = torch.load(deepaffinefile)
I_deepaffine = I_deepaffine.to('cuda')

stdfile = f'{prefix}deepaffinestd_{suffix}.h5'
if not os.path.isfile(stdfile): # standardize data
    import os
    affine_net = affine_net.to('cuda')
    eye = torch.eye(3).view(1,3,3).type(Iavg.dtype).to('cuda')
    with h5py.File(stdfile, 'w') as f:
        f.create_dataset('atlas', data=I_deepaffine.cpu().numpy())
        #f.create_dataset('A', data=As.cpu().numpy())
        #f.create_dataset('T', data=Ts.cpu().numpy())
        d = f.create_dataset('skullstripped', (len(oasis_ds), *I_deepaffine.shape[2:]), dtype=np.float32)
        with torch.no_grad():
            for ii in tqdm(range(len(oasis_ds))):
                i, J = oasis_ds[ii]
                J = J.unsqueeze(1).to('cuda')
                #A = As[i,...].to(Iaffine.device).unsqueeze(0)+eye
                #T = Ts[i,...].to(Iaffine.device).unsqueeze(0)
                A, T = affine_net(J)
                Ainv, Tinv = lm.affine_inverse(A+eye, T)
                Jdef = lm.affine_interp(J, Ainv, Tinv).cpu()
                d[i,...] = Jdef.numpy()
oasis_ds_std = OASISDataset(crop=None,
                        h5path=stdfile,
                        pooling=None,
                        one_scan_per_subject=False)

fluid_params = [.1,0,.01]
torch.save(fluid_params, f'fluidparams_{suffix}.pth')
convlddmmfile = f'{prefix}convlddmm_{suffix}.pth'
if not os.path.isfile(convlddmmfile): # conventional lddmm atlas
    print("Conventional LDDMM atlas building")
    res = lddmm_atlas(dataset=oasis_ds_std,
                                          I0=I_deepaffine.clone().to('cuda'),
                                          fluid_params=fluid_params,
                                          learning_rate_pose=1e-4,
                                          learning_rate_image=1e2,
                                          momentum_preconditioning=False,
                                          batch_size=20,
                                          num_epochs=500,
                                          gpu=args.local_rank,
                                          world_size=args.world_size,
                                          rank=rank)
    torch.save(res, convlddmmfile)
Ilddmm, mom_lddmm, epoch_losses, iter_losses = torch.load(convlddmmfile)

deeplddmmfile = f'{prefix}deeplddmm_{suffix}.pth'
if not os.path.isfile(deeplddmmfile): # deep lddmm atlas
    print("Deep LDDMM atlas building")
    res = \
        deep_lddmm_atlas(oasis_ds_std,
                     I=I_deepaffine.clone().to('cuda'),
                     #I=I_deeplddmm_ft.clone(),
                     num_epochs=500,
                     batch_size=10,
                     closed_form_image=False,
                     image_update_freq=100,
                     reg_weight=1e2,
                     #momentum_net=copy.deepcopy(mom_net_ft),
                     momentum_preconditioning=False,
                     learning_rate_pose=1e-6,
                     learning_rate_image=1e4,
                     fluid_params=fluid_params)
    torch.save(res, deeplddmmfile)
I_deeplddmm_ft, mom_net_ft, epoch_losses_deeplddmm_ft, iter_losses_deeplddmm_ft = torch.load(deeplddmmfile)
