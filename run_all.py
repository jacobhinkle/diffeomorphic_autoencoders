import torch
import numpy as np
import lagomorph
import os
import random

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

print(f"World size: {args.world_size} Local rank: {args.local_rank} Gpu: {gpu} Node Rank: {args.node_rank} NPN: {args.nprocs_per_node} Global Rank: {rank}")

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

# compute avg image
batch_size=12
avgfile = f'{prefix}oasisavg_{suffix}.pth'
if not os.path.isfile(avgfile):
    print("Voxel averaging")
    Iavg = batch_average(DataLoader(oasis_ds, batch_size=200, num_workers=8, pin_memory=True, shuffle=False), dim=0)
    if rank == 0: torch.save(Iavg, avgfile)
Iavg = torch.load(avgfile, map_location=loc)

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
    if rank == 0: torch.save(res, convaffinefile)
Iaffine, epoch_losses_affine, iter_losses_affine = torch.load(convaffinefile,
        map_location=loc)
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
    if rank == 0: torch.save(res, deepaffinefile)
I_deepaffine, affine_net, epoch_losses_deepaffine, full_losses_deepaffine, \
            iter_losses_deepaffine \
        = torch.load(deepaffinefile, map_location=loc)
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
Ilddmm, mom_lddmm, epoch_losses, iter_losses = torch.load(convlddmmfile,
        map_location=loc)
# On the test set, use same atlas-building code but with zero learning rate for
# the image
convlddmmtestfile = f'{prefix}convlddmm_test_{suffix}.pth'
if not os.path.isfile(convlddmmtestfile): # conventional lddmm atlas
    print("Conventional LDDMM Test")
    res = lddmm_atlas(dataset=oasis_ds_std,
                                          I0=Ilddmm.clone(),
                                          fluid_params=fluid_params,
                                          learning_rate_pose=1e-3,
                                          learning_rate_image=0e4,
                                          momentum_preconditioning=False,
                                          reg_weight=1e2,
                                          batch_size=30,
                                          num_epochs=500,
                                          gpu=gpu,
                                          world_size=args.world_size,
                                          rank=rank)
    if rank == 0: torch.save(res, convlddmmtestfile)
    del res
#Ilddmm, mom_lddmm, epoch_losses, iter_losses = torch.load(convlddmmtestfile,
        #map_location=loc)


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
                     fluid_params=fluid_params,
                     gpu=gpu,
                     world_size=args.world_size,
                     rank=rank)
    if rank == 0: torch.save(res, deeplddmmfile)
I_deeplddmm, mom_net, epoch_losses_deeplddmm, iter_losses_deeplddmm \
     = torch.load(deeplddmmfile, map_location=loc)

deeplddmmtestfile = f'{prefix}deeplddmm_test_{suffix}.pth'
if not os.path.isfile(deeplddmmtestfile): # deep lddmm atlas
    print("Deep LDDMM atlas building TEST")
    # manually do testing, just compute the regularization and image MSE here
    if world_size > 1:
        sampler = DistributedSampler(dataset, 
                num_replicas=args.world_size,
                rank=gpu)
    else:
        sampler = None
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
            num_workers=8, pin_memory=True, shuffle=False)
    with torch.no_grad():
        deeplddmm_test_loss = 0.0
        itbar = enumerate(dataloader)
        if rank == 0:
            itbar = tqdm(itbar, desc='minibatch')
        for it, (ix, img) in enumerate(itbar):
            img = img.to(I_deeplddmm.device)
            m = momentum_net(img)
            h = lm.expmap(metric, m, num_steps=lddmm_integration_steps)
            Idef = lm.interp(I_deeplddmm, h)
            v = metric.sharp(m)
            reg_term = (v*m).mean()
            loss = (mse_loss(Idef, img) + reg_weight*reg_term) \
                    * img.shape[0]/len(dataloader.dataset)
            if args.world_size > 1:
                all_reduce(loss)
                loss = loss/args.world_size
            deeplddmm_test_loss += loss.item()
    if rank == 0: torch.save((deeplddmm_test_loss,), deeplddmmtestfile)
