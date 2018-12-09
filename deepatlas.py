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

def MLP(widths, activation=None, last_layer_scale=1.0):
    if activation is None:
        activation = nn.ReLU()
    layers = []
    prev_size = widths[0]
    for sz in widths[1:-1]:
        layers.append(nn.Linear(prev_size, sz))
        layers.append(activation)
        prev_size = sz
    layers.append(nn.Linear(prev_size, widths[-1]))
    # skip activation on last layer
    # In the last layer, we multiply the random Kaiming initialization
    layers[-1].weight.data *= last_layer_scale
    layers[-1].bias.data *= last_layer_scale
    #default bias in last layer toward identity, not zero matrix
    #layers[-1].bias.data[0] += 1.0
    #layers[-1].bias.data[4] += 1.0
    #layers[-1].bias.data[8] += 1.0
    return nn.Sequential(*layers)


def conv_from_spec(conv_layers, image_size):
    layers = []
    nextchans = image_size[1]
    for conv_kernel_size, channel_growth, pool_stride in conv_layers:
        pool_kernel_size = pool_stride
        chans = nextchans
        nextchans *= channel_growth
        layers.append(nn.Conv3d(chans, nextchans, kernel_size=conv_kernel_size))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride))
    net = nn.Sequential(*layers)
    testim = torch.zeros(image_size)
    outim = net(testim)
    output_features = np.prod(outim.shape)
    return net, output_features

class AffinePredictorCNN(nn.Module):
    def __init__(self,
                 img_size=(1,1,256,256,256),
                 conv_layers=[(3,2,2),
                              (3,2,2),
                              (3,2,1),
                              (3,1,1),
                              (3,1,1),
                              (3,1,1)
                             ],
                 hidden=[256,64]):
        super(AffinePredictorCNN, self).__init__()
        self.img_size = img_size
        self.features, n_features = conv_from_spec(conv_layers, img_size)
        self.mlp = MLP([n_features] + hidden + [12],
                       last_layer_scale=1e-5)

    def forward(self, x):
        f = self.features(x.view(x.shape[0],*self.img_size[1:]))
        AT = self.mlp(f.view(f.shape[0],-1))
        eye = torch.eye(3).to(x.device).view(1,3,3)
        A = AT[:,:9].view(-1,3,3).contiguous()
        T = AT[:,9:].view(-1,3).contiguous()
        return A, T
    
def deep_affine_atlas(dataset,
        I=None,
        affine_net=None,
        num_epochs=500,
        batch_size=50,
        reg_weightA=1e1,
        reg_weightT=1e1,
        learning_rate_pose = 1e-3,
        learning_rate_image = 1e2,
        device='cuda'):
    from torch.utils.data import DataLoader, TensorDataset
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=8, pin_memory=True)
    if I is None:
        # initialize base image to mean
        I = batch_average(dataloader, dim=0)
    #else:
        #I = I.clone()
    I = I.to(device).view(1,1,*I.squeeze().shape)
    losses = []
    iter_losses = []
    full_losses = []
    if affine_net is None:
        affine_net = AffinePredictorCNN(img_size=I.shape)
    affine_net = affine_net.to(device)
    from torch.nn.functional import mse_loss
    I.requires_grad_(True)
    pose_optimizer = torch.optim.Adam(affine_net.parameters(),
                                      lr=learning_rate_pose,
                                      weight_decay=0e-5)
    image_optimizer = torch.optim.SGD([I],
                                      lr=learning_rate_image,
                                      weight_decay=0)
    print(f"Number of parameters: {sum([p.numel() for p in affine_net.features.parameters()])} {sum([p.numel() for p in affine_net.parameters()])}")
    eye = torch.eye(3).view(1,3,3).type(I.dtype).to(I.device)
    epbar = tqdm(range(num_epochs), desc='epoch', position=0)
    for epoch in epbar:
        epoch_loss = 0.0
        itbar = tqdm(dataloader, desc='iter', position=1)
        for it, (ix, img) in enumerate(itbar):
            pose_optimizer.zero_grad()
            image_optimizer.zero_grad()
            img = img.to(device)
            A, T = affine_net(img.view(img.shape[0],-1))
            Idef = lm.affine_interp(I, A+eye, T)
            regtermA = mse_loss(A,A)
            regtermT = mse_loss(T,T)
            loss = mse_loss(Idef, img) + .5*reg_weightA*regtermA + .5*reg_weightT*regtermT
            epoch_loss += loss.item()*img.shape[0]/len(dataloader.dataset)
            iter_losses.append(loss.item())
            itbar.set_postfix(minibatch_loss=loss.item())
            loss.backward()
            pose_optimizer.step()
            image_optimizer.step()
        losses.append(epoch_loss)
        epbar.set_postfix(epoch_loss=epoch_loss)
        if False: # track loss on entire dataset at every iteration
            with torch.no_grad():
                full_loss = 0.0
                for (img) in enumerate(dataloader):
                    img = img.to(device)
                    A, T = affine_net(img.view(img.shape[0],-1))
                    Idef = lm.affine_interp(I, A, T)
                    loss = mse_loss(Idef, J)*img.shape[0]/len(dataloader.dataset)
                    full_loss += loss.item()
                full_losses += full_loss
    return I.detach(), affine_net, losses, full_losses, iter_losses


def conv_down_from_spec(conv_layers, image_size):
    layers = []
    nextchans = image_size[1]
    for conv_kernel_size, channel_growth, pool_stride in conv_layers:
        pool_kernel_size = pool_stride
        chans = nextchans
        nextchans *= channel_growth
        layers.append(nn.Conv3d(chans, nextchans, kernel_size=conv_kernel_size))
        layers.append(nn.ReLU())
        if pool_stride > 1:
            layers.append(nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, return_indices=True))
    return layers

def conv_up_from_spec(conv_layers, image_size, out_channels):
    layers = []
    nextchans = image_size[1]*np.prod([cg for _,cg,_ in conv_layers]) # initial channels
    for i, (conv_kernel_size, channel_growth, pool_stride) in reversed(list(enumerate(conv_layers))):
        pool_kernel_size = pool_stride
        chans = nextchans
        if i == 0:
            nextchans = out_channels
        else:
            nextchans //= channel_growth
        if pool_stride > 1:
            layers.append(nn.MaxUnpool3d(kernel_size=pool_kernel_size, stride=pool_stride))
        layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose3d(chans, nextchans, kernel_size=conv_kernel_size))
    return layers


class PreconditionedExpMapFunction(torch.autograd.Function):
    """This is the same as lm.expmap, but with a preconditioned gradient on the
    momentum, obtained by applying metric.flat to the result of backward()."""
    @staticmethod
    def forward(ctx, metric, m, num_steps):
        ctx.save_for_backward(m)
        ctx.metric = metric
        ctx.h = lm.expmap(metric, m, num_steps=num_steps)
        return ctx.h
    @staticmethod
    def backward(ctx, gradout):
        ctx.h.backward()
        m = ctx.saved_tensors
        dm = ctx.metric.flat(m.grad)
        return _, dm, _
precond_expmap = PreconditionedExpMapFunction.apply


class MomentumPredictor(nn.Module):
    def __init__(self,
                 img_size=(1,1,256,256,256),
                 conv_layers=[(7,8,2),
                              (3,4,2),
                              (3,4,2),
                              (5,2,1)]):
        super(MomentumPredictor, self).__init__()
        self.img_size = img_size
        self.down_layers = conv_down_from_spec(conv_layers, img_size)
        from itertools import chain
        self.down_layers_params = nn.ParameterList(chain(*[p.parameters() \
                                             for p in self.down_layers]))
        self.up_layers = conv_up_from_spec(conv_layers, img_size, 3)
        self.up_layers_params = nn.ParameterList(chain(*[p.parameters() \
                                             for p in self.up_layers]))
        last_layer_scale=1e-5
        with torch.no_grad():
            self.up_layers[-1].weight.mul_(last_layer_scale)
            self.up_layers[-1].bias.mul_(last_layer_scale)
    def forward(self, x):
        d = x
        inds = []
        szs = []
        for l in self.down_layers:
            szs.append(d.shape)
            ix = None
            if isinstance(l, nn.MaxPool3d):
                d, ix = l(d)
            else:
                d = l(d)
            inds.append(ix)
        for l, ix, sz in zip(self.up_layers, reversed(inds), reversed(szs)):
            if isinstance(l, nn.MaxUnpool3d):
                d = l(d, ix, output_size=sz)
            elif isinstance(l, nn.ConvTranspose3d):
                d = l(d, output_size=sz)
            else:
                d = l(d)
        return d

    
def deep_lddmm_atlas(dataset,
        I,
        fluid_params=[1e-1,0.,.01],
        num_epochs=500,
        batch_size=2,
        reg_weight=.001,
        momentum_net=None,
        learning_rate_pose=1e-5,
        learning_rate_image=1e6):
    from torch.utils.data import DataLoader, TensorDataset
    #I = I.clone()
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=8, pin_memory=True)
    epoch_losses = []
    iter_losses = []
    full_losses = []
    if momentum_net is None:
        momentum_net = MomentumPredictor(img_size=I.shape)
    momentum_net = momentum_net.to(I.device)
    from torch.nn.functional import mse_loss
    pose_optimizer = torch.optim.Adam(momentum_net.parameters(),
                                      lr=learning_rate_pose,
                                      weight_decay=1e-3)
    metric = lm.FluidMetric(fluid_params)
    epbar = tqdm(range(num_epochs), position=0)
    for epoch in epbar:
        epoch_loss = 0.0
        iterbar = tqdm(dataloader, position=1)
        for it, (ix, img) in enumerate(iterbar):
            if I.grad is not None:
                I.grad.detach_()
                I.grad.zero_()
            I.requires_grad_(True)
            pose_optimizer.zero_grad()
            img = img.to(I.device)
            m = momentum_net(img)
            h = lm.expmap(metric, m, num_steps=5)
            Idef = lm.interp(I, h)
            reg_term = (metric.sharp(m)*m).mean()
            loss = mse_loss(Idef, img) + reg_weight*reg_term
            epoch_loss += loss.item()*img.shape[0]/len(dataset)
            iter_losses.append(loss.item())
            iterbar.set_postfix(minibatch_loss=loss.item())
            loss.backward()
            with torch.no_grad():
                I.add_(-learning_rate_image, I.grad)
            pose_optimizer.step()
        epoch_losses.append(epoch_loss)
        epbar.set_postfix(epoch_loss=epoch_loss)
        if False:
            with torch.no_grad():
                A, T = affine_net(J.view(J.shape[0],-1))
                Idef = lm.affine_interp(I, A, T)
                full_loss = mse_loss(Idef, J)
                full_losses.append(full_loss)
    return I.detach(), momentum_net, epoch_losses, iter_losses, full_losses


