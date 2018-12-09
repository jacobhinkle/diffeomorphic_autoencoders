import torch
from torch import nn
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
                batch_size=5,
                affine_steps=100,
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
        itbar = tqdm(dataloader, desc='iter', position=1)
        for it, (ix, img) in enumerate(itbar):
            A = As[ix,...].contiguous().to(device)
            T = Ts[ix,...].contiguous().to(device)
            img = img.to(device)
            img.requires_grad_(False)
            A, T, losses_match = affine_matching(I,
                                                 img,
                                                 A=A,
                                                 T=T,
                                                 affine_steps=affine_steps,
                                                 reg_weightA=reg_weightA,
                                                 reg_weightT=reg_weightT,
                                                 learning_rate_A=learning_rate_A,
                                                 learning_rate_T=learning_rate_T,
                                                 progress_bar=False)
            A.requires_grad_(False)
            T.requires_grad_(False)
            I.requires_grad_(True)
            image_optimizer.zero_grad()
            Idef = lm.affine_interp(I, A+eye, T)
            loss = mse_loss(Idef, img)
            loss.backward()
            image_optimizer.step()
            loss.detach_()
            with torch.no_grad():
                regtermA = mse_loss(A,A)
                regtermT = mse_loss(T,T)
                regloss = .5*reg_weightA*regtermA + .5*reg_weightT*regtermT
            itloss = loss.item() + regloss.item()
            epoch_loss += itloss*img.shape[0]/len(dataloader.dataset)
            iter_losses.append(itloss)
            itbar.set_postfix(minibatch_loss=itloss)
            As[ix,...] = A.detach().to(As.device)
            Ts[ix,...] = T.detach().to(Ts.device)
        losses.append(epoch_loss)
        epbar.set_postfix(epoch_loss=epoch_loss)
    return I.detach(), losses, iter_losses

def lddmm_atlas(dataset,
        I=None,
        num_epochs=500,
        batch_size=2,
        lddmm_steps=10,
        lddmm_integration_steps=5,
        reg_weight=1e-3,
        learning_rate_pose = 1e-4,
        learning_rate_image = 1e1,
        fluid_params=[1.0,.1,.01],
        device='cuda',
        momentum_pattern='oasis_momenta/momentum_{}.pth'):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)
    if I is None:
        # initialize base image to mean
        I = batch_average(dataloader, dim=0)
    I = I.to(device).view(1,1,*I.squeeze().shape)
    image_optimizer = torch.optim.SGD([I],
                                      lr=learning_rate_image,
                                      weight_decay=0)
    metric = lm.FluidMetric(fluid_params)
    losses = []
    iter_losses = []
    epbar = tqdm(range(num_epochs), desc='epoch', position=0)
    for epoch in epbar:
        epoch_loss = 0.0
        itbar = tqdm(dataloader, desc='iter', position=1)
        for it, (ix, img) in enumerate(itbar):
            m = torch.zeros(img.shape[0],3,*I.shape[-3:], dtype=I.dtype)
            for i,ii in enumerate(ix):
                try:
                    m[i,...] = torch.load(momentum_pattern.format(ii))
                except FileNotFoundError:
                    pass
            m = m.to(device)
            img = img.to(device)
            I.requires_grad_(False)
            img.requires_grad_(False)
            m, losses_match = lddmm_matching(I, img, m=m, lddmm_steps=lddmm_steps, progress_bar=False)
            m.requires_grad_(False)
            I.requires_grad_(True)
            h = lm.expmap(metric, m, num_steps=lddmm_integration_steps)
            Idef = lm.interp(I, m)
            v = metric.sharp(m)
            regterm = (v*m).mean()
            loss = mse_loss(Idef, img) + reg_weight*regterm
            image_optimizer.step()
            epoch_loss += loss.item()*img.shape[0]/len(dataloader.dataset)
            iter_losses.append(loss.item())
            itbar.set_postfix(minibatch_loss=loss.item())
            for i,ii in enumerate(ix):
                torch.save(m[i,...], momentum_pattern.format(ii))
        losses.append(epoch_loss)
        epbar.set_postfix(epoch_loss=epoch_loss)
    return I, losses, iter_losses

