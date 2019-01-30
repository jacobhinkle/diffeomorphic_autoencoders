import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader
import h5py
from utils import tqdm
import numpy as np
import random

from matching import *

import lagomorph as lm

def affine_atlas(dataset,
                 As,
                 Ts,
                I=None,
                num_epochs=1000,
                batch_size=50,
                affine_steps=1,
                reg_weightA=0e1,
                reg_weightT=0e1,
                learning_rate_A=1e-3,
                learning_rate_T=1e-2,
                learning_rate_I=1e5,
                loader_workers=8,
                device='cuda'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
            num_workers=loader_workers, pin_memory=True)
    if I is None:
        # initialize base image to mean
        I = batch_average(dataloader, dim=0)
    else:
        I = I.clone()
    I = I.to(device).view(1,1,*I.squeeze().shape)
    image_optimizer = torch.optim.SGD([I],
                                      lr=learning_rate_I,
                                      weight_decay=0.)
    eye = torch.eye(3).view(1,3,3).type(I.dtype).to(I.device)
    losses = []
    iter_losses = []
    epbar = tqdm(range(num_epochs), desc='epoch', position=0)
    for epoch in epbar:
        epoch_loss = 0.0
        itbar = dataloader
        #itbar = tqdm(dataloader, desc='iter', position=1)
        image_optimizer.zero_grad()
        for it, (ix, img) in enumerate(itbar):
            A = As[ix,...].detach().to(device)
            T = Ts[ix,...].detach().to(device)
            img = img.to(device)
            img.requires_grad_(False)
            #A, T, losses_match = affine_matching(I,
            #                                     img,
            #                                     A=A,
            #                                     T=T,
            #                                     affine_steps=affine_steps,
            #                                     reg_weightA=reg_weightA,
            #                                     reg_weightT=reg_weightT,
            #                                     learning_rate_A=learning_rate_A,
            #                                     learning_rate_T=learning_rate_T,
            #                                     progress_bar=False)
            A.requires_grad_(True)
            T.requires_grad_(True)
            if A.grad is not None:
                A.grad.zero_()
            if T.grad is not None:
                T.grad.zero_()
            I.requires_grad_(True)
            Idef = lm.affine_interp(I, A+eye, T)
            regtermA = mse_loss(A,A)
            regtermT = mse_loss(T,T)
            regloss = .5*reg_weightA*regtermA + .5*reg_weightT*regtermT
            # average over entire dataset for loss, since image updates once per epoch
            loss = (mse_loss(Idef, img)+regloss)*img.shape[0]/len(dataloader.dataset)
            loss.backward()
            loss.detach_()
            epoch_loss += loss.item()
            iter_losses.append(loss)
            with torch.no_grad():
                A.add_(-learning_rate_A*len(dataloader.dataset), A.grad)
                T.add_(-learning_rate_T*len(dataloader.dataset), T.grad)
            #itbar.set_postfix(minibatch_loss=itloss)
            As[ix,...] = A.detach().to(As.device)
            Ts[ix,...] = T.detach().to(Ts.device)
        image_optimizer.step()
        losses.append(epoch_loss)
        epbar.set_postfix(epoch_loss=epoch_loss)
    return I.detach(), losses, iter_losses

class DenseInterp(nn.Module):
    """Very simple module wrapper that enables simple data parallel"""
    def __init__(self, I0):
        super(DenseInterp, self).__init__()
        self.I = nn.Parameter(I0)
    def forward(self, h):
        return lm.interp(self.I, h)

def lddmm_atlas(dataset,
        I0=None,
        num_epochs=500,
        batch_size=10,
        lddmm_steps=1,
        lddmm_integration_steps=5,
        reg_weight=1e2,
        learning_rate_pose = 2e2,
        learning_rate_image = 1e4,
        fluid_params=[0.1,0.,.01],
        device='cuda',
        momentum_preconditioning=True,
        momentum_pattern='oasis_momenta/momentum_{}.pth',
        gpu=None,
        world_size=1,
        rank=1):
    if world_size > 1:
        sampler = DistributedSampler(dataset, 
                num_replicas=world_size,
                rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
            num_workers=8, pin_memory=True, shuffle=False)
    if I0 is None:
        # initialize base image to mean
        I0 = batch_average(dataloader, dim=0)
    I0 = I0.view(1,1,*I0.squeeze().shape)
    I = DenseInterp(I0)
    if gpu is not None:
        I = DistributedDataParallel(I, device_ids=[gpu], output_device=gpu)
        I = I.to(f'cuda:{gpu}')
    else:
        I = I.to(device)
    image_optimizer = torch.optim.SGD(I.parameters(),
                                      lr=learning_rate_image,
                                      weight_decay=0)
    metric = lm.FluidMetric(fluid_params)
    losses = []
    iter_losses = []
    epbar = tqdm(range(num_epochs), desc='epoch')
    ms = torch.zeros(len(dataset),3,*I0.shape[-3:], dtype=I0.dtype).pin_memory()
    for epoch in epbar:
        epoch_loss = 0.0
        itbar = dataloader
        image_optimizer.zero_grad()
        for it, (ix, img) in enumerate(itbar):
            m = ms[ix,...].detach()
            m = m.to(device)
            img = img.to(device)
            m.requires_grad_(True)
            if m.grad is not None:
                m.grad.zero_()
            h = lm.expmap(metric, m, num_steps=lddmm_integration_steps)
            Idef = I(h)
            v = metric.sharp(m)
            regterm = (v*m).mean()
            loss = (mse_loss(Idef, img) + reg_weight*regterm)*img.shape[0]/len(dataloader.dataset)
            loss.backward()
            with torch.no_grad():
                p = m.grad
                if momentum_preconditioning:
                    p = metric.flat(p)
                m.add_(-learning_rate_pose*len(dataloader.dataset), p)
                del p
            epoch_loss += loss.item()
            iter_losses.append(loss.item())
            #itbar.set_postfix(minibatch_loss=loss.item())
            if False:
                for i,ii in enumerate(ix):
                    torch.save(m[i,...], momentum_pattern.format(ii))
            else:
                ms[ix,...] = m.detach().cpu()
        image_optimizer.step()
        losses.append(epoch_loss)
        epbar.set_postfix(epoch_loss=epoch_loss)
    return I.I.detach(), ms.detach(), losses, iter_losses

