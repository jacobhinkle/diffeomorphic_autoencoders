import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader
from utils import tqdm
import h5py
import numpy as np
import random

import lagomorph as lm

def affine_matching(I,
                    J,
                    A=None,
                    T=None,
                    affine_steps=100,
                    reg_weightA=1e2,
                    reg_weightT=1e1,
                    learning_rate_A=1e-4,
                    learning_rate_T=1e-2,
                    progress_bar=True):
    """Matching image I to J via affine transform"""
    if A is None:
        A = torch.zeros((I.shape[0],3,3), dtype=I.dtype).to(I.device)
    if T is None:
        T = torch.zeros((I.shape[0],3), dtype=I.dtype).to(I.device)
    J = J.to(I.device)
    losses = []
    I.requires_grad_(False)
    J.requires_grad_(False)
    steps = range(affine_steps)
    eye = torch.eye(3).view(1,3,3).type(I.dtype).to(I.device)
    if progress_bar: steps = tqdm(steps)
    for mit in steps:
        A.requires_grad_(True)
        T.requires_grad_(True)
        if A.grad is not None and T.grad is not None:
            A.grad.detach_()
            A.grad.zero_()
            T.grad.detach_()
            T.grad.zero_()
        Idef = lm.affine_interp(I, A+eye, T)
        regtermA = mse_loss(A,A)
        regtermT = mse_loss(T,T)
        loss = mse_loss(Idef, J) + .5*reg_weightA*regtermA + .5*reg_weightT*regtermT
        loss.backward()
        loss.detach_()
        with torch.no_grad():
            losses.append(loss)
            #if torch.isnan(losses[-1]).item():
                #print(f"loss is NaN at iter {mit}")
                #break
            #if mit > 0 and losses[-1].item() > losses[-2].item():
                #print(f"loss increased at iter {mit}")
            A.add_(-learning_rate_A, A.grad)
            T.add_(-learning_rate_T, T.grad)
    return A.detach(), T.detach(), [l.item() for l in losses]

def lddmm_matching( I,
                    J,
                    m=None,
                    lddmm_steps=1000,
                    lddmm_integration_steps=10,
                    reg_weight=1e-1,
                    diffeo_scale=None,
                    learning_rate_pose = 2e-2,
                    fluid_params=[1.0,.1,.01],
                    progress_bar=True
                  ):
    """Matching image I to J via LDDMM"""
    if diffeo_scale is not None:
        defsh = [I.shape[0], 3] + [s//diffeo_scale for s in I.shape[2:]]
    else:
        defsh = [I.shape[0], 3] + list(I.shape[2:])
    if m is None:
        m = torch.zeros(defsh, dtype=I.dtype).to(I.device)
    else:
        assert m.shape != defsh
    J = J.to(I.device)
    losses = []
    metric = lm.FluidMetric(fluid_params)
    m.requires_grad_()
    pb = range(lddmm_steps)
    if progress_bar: pb = tqdm(pb)
    for mit in pb:
        if m.grad is not None:
            m.grad.detach_()
            m.grad.zero_()
        m.requires_grad_()
        h = lm.expmap(metric, m, num_steps=lddmm_integration_steps)
        if diffeo_scale is not None:
            h = lm.regrid(h, shape=I.shape[2:], displacement=True)
        Idef = lm.interp(I, h)
        loss = mse_loss(Idef, J)
        loss.backward()
        loss.detach_()
        with torch.no_grad():
            p = metric.flat(m.grad).detach()
            v = metric.sharp(m)
            regterm = (v*m).sum().detach()
            losses.append(loss.detach()+ .5*reg_weight*regterm)
            if torch.isnan(losses[-1]).item():
                print(f"loss is NaN at iter {mit}")
                break
            if mit > 0 and losses[-1].item() > losses[-2].item():
                print(f"loss increased at iter {mit}")
            p.add_(reg_weight, m)
            m.add_(-learning_rate_pose, p)
    return m.detach(), [l.item() for l in losses]
