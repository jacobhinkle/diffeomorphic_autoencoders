import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import all_reduce
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

def L2norm(a):
    aflat = a.view(-1)
    return torch.dot(aflat, aflat)
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
                gpu=None,
                world_size=1,
                rank=0):
    from torch.utils.data import DataLoader, TensorDataset
    if world_size > 1:
        sampler = DistributedSampler(dataset, 
                num_replicas=world_size,
                rank=rank)
    else:
        sampler = None
    if gpu is None:
        device = 'cpu'
    else:
        device = f'cuda:{gpu}'
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            shuffle=False, num_workers=8, pin_memory=True)
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
    epbar = range(num_epochs)
    if rank == 0:
        epbar = tqdm(epbar, desc='epoch', position=0)
    for epoch in epbar:
        epoch_loss = 0.0
        itbar = dataloader
        #itbar = tqdm(dataloader, desc='iter', position=1)
        image_optimizer.zero_grad()
        for it, (ix, img) in enumerate(itbar):
            A = As[ix,...].detach().to(device).contiguous()
            T = Ts[ix,...].detach().to(device).contiguous()
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
            for affit in range(affine_steps):
                A.requires_grad_(True)
                T.requires_grad_(True)
                if A.grad is not None:
                    A.grad.detach_()
                    A.grad.zero_()
                if T.grad is not None:
                    T.grad.detach_()
                    T.grad.zero_()
                I.requires_grad_(True)
                Idef = lm.affine_interp(I, A+eye, T)
                regloss = 0
                if reg_weightA > 0:
                    regtermA = L2norm(A)
                    regloss = regloss + .5*reg_weightA*regtermA
                if reg_weightT > 0:
                    regtermT = L2norm(T)
                    regloss = regloss + .5*reg_weightT*regtermT
                loss = (mse_loss(Idef, img, reduction='sum')*(1./np.prod(I.shape[2:])) + regloss) \
                        / (img.shape[0])
                loss.backward()
                loss.detach_()
                iter_losses.append(loss)
                with torch.no_grad():
                    li = (loss*(img.shape[0]/len(dataloader.dataset))).detach()
                    iter_losses.append(li.item())
                    A.add_(-learning_rate_A, A.grad)
                    T.add_(-learning_rate_T, T.grad)
            with torch.no_grad():
                li = (loss*(img.shape[0]/len(dataloader.dataset))).detach()
                epoch_loss = epoch_loss + li
            #itbar.set_postfix(minibatch_loss=itloss)
            As[ix,...] = A.detach().to(As.device)
            Ts[ix,...] = T.detach().to(Ts.device)
        with torch.no_grad():
            if world_size > 1:
                all_reduce(epoch_loss)
                all_reduce(I.grad)
                I.grad = I.grad/(len(dataloader)*world_size)
        image_optimizer.step()
        losses.append(epoch_loss.item())
        if rank == 0: epbar.set_postfix(epoch_loss=epoch_loss.item())
    return I.detach(), As, Ts, losses, iter_losses

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
        rank=0):
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
    #I = DenseInterp(I0)
    #if gpu is not None:
        #I = DistributedDataParallel(I, device_ids=[gpu], output_device=gpu)
        #I = I.to(f'cuda:{gpu}')
    #else:
    I = I0.clone()
    I = I.to(device)
    #image_optimizer = torch.optim.SGD(I.parameters(),
    image_optimizer = torch.optim.SGD([I],
                                      lr=learning_rate_image,
                                      weight_decay=0)
    metric = lm.FluidMetric(fluid_params)
    losses = []
    reg_terms = []
    iter_losses = []
    epbar = range(num_epochs)
    if rank == 0:
        epbar = tqdm(epbar, desc='epoch')
    ms = torch.zeros(len(dataset),3,*I0.shape[-3:], dtype=I0.dtype).pin_memory()
    for epoch in epbar:
        epoch_loss = 0.0
        epoch_reg_term = 0.0
        itbar = dataloader
        I.requires_grad_(True)
        image_optimizer.zero_grad()
        for it, (ix, img) in enumerate(itbar):
            m = ms[ix,...].detach()
            m = m.to(device)
            img = img.to(device)
            for lit in range(lddmm_steps):
                # compute image gradient in last step
                I.requires_grad_(lit == lddmm_steps - 1)
                # enables taking multiple LDDMM step per image update
                m.requires_grad_(True)
                if m.grad is not None:
                    m.grad.detach_()
                    m.grad.zero_()
                h = lm.expmap(metric, m, num_steps=lddmm_integration_steps)
                #Idef = I(h)
                Idef = lm.interp(I, h)
                v = metric.sharp(m)
                regterm = reg_weight*(v*m).sum()
                loss = (mse_loss(Idef, img, reduction='sum') + regterm) \
                        / (img.numel())
                loss.backward()
                # this makes it so that we can reduce the loss and eventually get
                # an accurate MSE for the entire dataset
                with torch.no_grad():
                    li = (loss*(img.shape[0]/len(dataloader.dataset))).detach()
                    p = m.grad
                    if momentum_preconditioning:
                        p = metric.flat(p)
                    m.add_(-learning_rate_pose, p)
                    if world_size > 1:
                        all_reduce(li)
                    iter_losses.append(li.item())
                    m = m.detach()
                    del p
            with torch.no_grad():
                epoch_loss += li
                ri = (regterm*(img.shape[0]/(img.numel()*len(dataloader.dataset)))).detach()
                epoch_reg_term += ri
                ms[ix,...] = m.detach().cpu()
            del m, h, Idef, v, loss, regterm, img
        with torch.no_grad():
            if world_size > 1:
                all_reduce(epoch_loss)
                all_reduce(epoch_reg_term)
                all_reduce(I.grad)
                I.grad = I.grad/world_size
            # average over iterations
            I.grad = I.grad / len(dataloader)
        image_optimizer.step()
        losses.append(epoch_loss.item())
        reg_terms.append(epoch_reg_term.item())
        if rank == 0:
            epbar.set_postfix(epoch_loss=epoch_loss.item(),
                    epoch_reg=epoch_reg_term.item())
    #return I.state_dict()['I'].detach(), ms.detach(), losses, iter_losses
    return I.detach(), ms.detach(), losses, iter_losses


