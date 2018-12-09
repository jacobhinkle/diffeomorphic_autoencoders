import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm
import numpy as np
import random

import lagomorph as lm
from oasis_data import *
from matching import *
from atlas import *
from deepatlas import *


def batch_average(dataloader, **kwargs):
    """Compute the average using streaming batches from a dataloader along a given dimension"""
    avg = None
    sumsizes = 0
    for (i, img) in tqdm(dataloader, 'image avg'):
        sz = img.shape[0]
        avi = img.to('cuda').mean(**kwargs)
        if avg is None:
            avg = avi
        else:
            # add similar-sized numbers using this running average
            avg = avg*(sumsizes/(sumsizes+sz)) + avi*(sz/(sumsizes+sz))
        sumsizes += sz
    return avg


if __name__ == '__main__':

    if args.debug:
        lm.set_debug_mode(True)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    docrop = 0
    if args.docrop:
	crop = np.asarray(
	[[ 47, 212],
	 [ 16, 251],
	 [  9, 228]])
	if True:
	    crop[:,0] -= 1
	    crop[:,1] += 1
    else:
	crop = None
    ds_first = 100
    ds_pooling = 2
    oasis_ds = OASISDataset(crop=crop, first=ds_first, pooling=ds_pooling)

    batch_size=12
    sz = oasis_ds[0][1].shape[2]
    l = len(oasis_ds)
    if docrop:
	Ifile = f'oasisavg_crop_{sz}_{l}.pth'
    else:
	Ifile = f'oasisavg_nocrop_{sz}_{l}.pth'
    try:
	Iavg = torch.load(Ifile)
    except FileNotFoundError:
	Iavg = batch_average(DataLoader(oasis_ds, batch_size=200, num_workers=8, pin_memory=True, shuffle=False), dim=0)
	torch.save(Iavg, Ifile)
    I = Iavg
    plt.imshow(I.cpu().numpy()[:,:,I.shape[3]//2].squeeze(), cmap='gray')
    plt.title('Average image');

    _, I = oasis_ds[0]
    _, J = oasis_ds[10]
    I = I.unsqueeze(1).to('cuda')
    J = J.unsqueeze(1).to(I.device)
    I.requires_grad_(False)
    J.requires_grad_(False)
    Amatch, Tmatch, losses_affine_match = affine_matching(I, J)

    if args.plot:
        sl = I.shape[3]//2
        Idef = lm.affine_interp(I, Amatch+eye, Tmatch)
        plt.plot(losses_affine_match, '-')
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(   I[0,0,:,sl,:].cpu().numpy().squeeze(), cmap='gray')
        plt.subplot(1,3,2)
        plt.imshow(Idef[0,0,:,sl,:].cpu().numpy().squeeze(), cmap='gray')
        plt.subplot(1,3,3)
        plt.imshow(   J[0,0,:,sl,:].cpu().numpy().squeeze(), cmap='gray')

    As = torch.zeros((len(oasis_ds),3,3), dtype=torch.float32).to('cuda')
    Ts = torch.zeros((len(oasis_ds),3), dtype=torch.float32).to('cuda')
    Iaffine, losses, iter_losses = affine_atlas(
	    dataset=OASISDataset(crop=crop, first=ds_first, pooling=ds_pooling),
	    I=Iavg, As=As, Ts=Ts, num_epochs=50)

    if args.plot:
        sl = I.shape[3]//2
        plt.plot(losses)
        plt.figure()
        plt.plot(iter_losses)
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(   Iavg[0,:,sl,:].cpu().numpy().squeeze(), cmap='gray')
        plt.title('Average')
        plt.subplot(1,2,2)
        plt.imshow(   Iaffine[0,0,:,sl,:].cpu().numpy().squeeze(), cmap='gray')
        plt.title('Affine Atlas');

    stdfile = f'oasis_affine_std_crop{docrop}_{sz}_{l}.h5'
    import os
    if os.path.isfile(stdfile):
	oasis_ds_std = OASISDataset(h5path=stdfile)
	with h5py.File(stdfile, 'r') as f:
	    As = torch.Tensor(f['A']).to('cuda')
	    Ts = torch.Tensor(f['T']).to('cuda')
    else:
	with h5py.File(stdfile, 'w') as f:
	    f.create_dataset('atlas', data=Iaffine.cpu().numpy())
	    f.create_dataset('A', data=As.cpu().numpy())
	    f.create_dataset('T', data=Ts.cpu().numpy())
	    d = f.create_dataset('skullstripped', (len(oasis_ds), *Iaffine.shape[2:]), dtype=np.float32)
	    with torch.no_grad():
		for ii in tqdm(range(len(oasis_ds))):
		    i, J = oasis_ds[ii]
		    J = J.unsqueeze(1).to(Iaffine.device)
		    A = As[i,...].to(Iaffine.device).unsqueeze(0)+eye
		    T = Ts[i,...].to(Iaffine.device).unsqueeze(0)
		    Ainv, Tinv = lm.affine_inverse(A, T)
		    Idef = lm.affine_interp(J, Ainv, Tinv).cpu()
		    d[i,...] = Idef.numpy()
	    oasis_ds_std = OASISDataset(h5path=stdfile)

    _, I = oasis_ds_std[0]
    _, J = oasis_ds_std[10]
    I = I.unsqueeze(1).to('cuda')
    J = J.unsqueeze(1).to(I.device)
    I.requires_grad_(False)
    J.requires_grad_(False)
    #fluid_params=[1e-2,.0,.01]
    fluid_params=[5e-2,.0,.01]
    diffeo_scale=None
    mmatch, losses_match = lddmm_matching(I, J, fluid_params=fluid_params, diffeo_scale=diffeo_scale)
    if args.plot:
        sl = I.shape[3]//2
        metric = lm.FluidMetric(fluid_params)
        hsmall = lm.expmap(metric, mmatch, num_steps=10)
        h = hsmall
        if diffeo_scale is not None:
            h = lm.regrid(h, shape=I.shape[2:], displacement=True)
        Idef = lm.interp(I, h)
        plt.plot(losses_match, '-o')
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(   I[0,0,:,sl,:].cpu().numpy().squeeze(), cmap='gray')
        plt.subplot(1,3,2)
        plt.imshow(Idef[0,0,:,sl,:].cpu().numpy().squeeze(), cmap='gray')
        plt.subplot(1,3,3)
        plt.imshow(   J[0,0,:,sl,:].cpu().numpy().squeeze(), cmap='gray')
        plt.figure()
        if diffeo_scale is None:
            ds = 1
        else:
            ds = diffeo_scale
        lm.quiver(mmatch[:,:2,:,sl//ds,:]/fluid_params[2])
        plt.figure()
        lm.gridplot(hsmall[:,:2,:,sl//ds,:], 32, 32)
        plt.title('Natural scale deformation')
        plt.figure()
        lm.gridplot(h[:,:2,:,sl,:])
        plt.title('Regridded Deformation');

    Ilddmm, losses, iter_losses = lddmm_atlas(dataset=oasis_ds_std, I=I, num_epochs=5)
    if args.plot:
        sl = I.shape[3]//2
        plt.plot(iter_losses)
        plt.figure()
        plt.imshow(Ilddmm.detach()[0,0,:,sl,:].cpu().numpy().squeeze(), cmap='gray')
        plt.title('LDDMM atlas')

    I_deeplddmm, mom_net, epoch_losses, full_losses  = \
	deep_lddmm_atlas(oasis_ds_std, I=Iaffine, num_epochs=100)

    if args.plot:
        plt.plot(epoch_losses)
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(   Iavg[0,:,sl,:].cpu().numpy().squeeze(), cmap='gray')
        plt.title('Average')
        plt.subplot(1,3,2)
        plt.imshow(   Iaffine.detach()[0,0,:,sl,:].cpu().numpy().squeeze(), cmap='gray')
        plt.title('Affine Atlas');
        plt.subplot(1,3,3)
        plt.imshow(   I_deeplddmm[0,0,:,sl,:].cpu().numpy().squeeze(), cmap='gray')
        plt.title('Deep LDDMM Atlas');
