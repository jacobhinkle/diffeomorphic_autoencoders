import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import all_reduce
from torch.utils.data.distributed import DistributedSampler
import h5py
from utils import tqdm
import numpy as np
import random

from matching import *

from importlib import reload
import lagomorph
reload(lagomorph)
import lagomorph as lm

def MLP(widths, activation=None, last_layer_scale=1.0, dropout=None):
    if activation is None:
        activation = nn.ReLU()
    layers = []
    prev_size = widths[0]
    for sz in widths[1:-1]:
        if dropout is not None:
            layers.append(nn.Dropout(p=dropout))
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
                              (3,1,1)],
                 hidden=[256,64],
                 dropout=.5):
        super(AffinePredictorCNN, self).__init__()
        self.img_size = img_size
        self.features, n_features = conv_from_spec(conv_layers, img_size)
        self.mlp = MLP([n_features] + hidden + [12],
                       last_layer_scale=1e-5,
                       dropout=dropout)

    def forward(self, x, eye=None):
        f = self.features(x.view(x.shape[0],*self.img_size[1:]))
        AT = self.mlp(f.view(f.shape[0],-1))
        A = AT[:,:9].view(-1,3,3).contiguous()
        T = AT[:,9:].view(-1,3).contiguous()
        return A, T
    
def L2norm(a):
    aflat = a.view(-1)
    return torch.dot(aflat, aflat)

def deep_affine_atlas(dataset,
        I=None,
        affine_net=None,
        num_epochs=500,
        batch_size=50,
        reg_weightA=1e1,
        reg_weightT=1e1,
        dropout=.5,
        learning_rate_pose = 1e-3,
        learning_rate_image = 1e2,
        test_dataset = None,
        test_every = 10,
        gpu=None,
        world_size=1,
        rank=0):
    from torch.utils.data import DataLoader, TensorDataset
    if world_size > 1:
        sampler = DistributedSampler(dataset, 
                num_replicas=world_size,
                rank=rank)
        if test_dataset is not None:
            test_sampler = DistributedSampler(test_dataset,
                    num_replicas=world_size,
                    rank=rank)
        else:
            test_dataset = None
    else:
        sampler = None
        test_sampler = None
    if gpu is None:
        device = 'cpu'
    else:
        device = f'cuda:{gpu}'
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            shuffle=False, num_workers=8, pin_memory=True)
    if test_dataset is not None:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                            shuffle=False, num_workers=8, pin_memory=True)

    if I is None:
        # initialize base image to mean
        I = batch_average(dataloader, dim=0)
    #else:
        #I = I.clone()
    I = I.to(device).view(1,1,*I.shape[-3:])
    losses = []
    iter_losses = []
    full_losses = []
    test_losses = []
    test_loss = None
    if affine_net is None:
        affine_net = AffinePredictorCNN(img_size=I.shape, dropout=dropout)
    affine_net = affine_net.to(device)
    if world_size > 1:
        affine_net = DistributedDataParallel(affine_net,
                device_ids=[gpu], output_device=gpu)
    from torch.nn.functional import mse_loss
    I.requires_grad_(True)
    pose_optimizer = torch.optim.Adam(affine_net.parameters(),
                                      lr=learning_rate_pose,
                                      weight_decay=1e-3)
    image_optimizer = torch.optim.SGD([I],
                                      lr=learning_rate_image,
                                      weight_decay=0)
    #print(f"Number of parameters: {sum([p.numel() for p in affine_net.features.parameters()])} {sum([p.numel() for p in affine_net.parameters()])}")
    eye = torch.eye(3).view(1,3,3).type(I.dtype).to(I.device)
    epbar = range(num_epochs)
    if rank == 0:
        epbar = tqdm(epbar, desc='epoch')
    for epoch in epbar:
        epoch_loss = 0.0
        itbar = dataloader
        image_optimizer.zero_grad()
        for it, (ix, img) in enumerate(itbar):
            pose_optimizer.zero_grad()
            img = img.to(device)
            A, T = affine_net(img.view(img.shape[0],-1))
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
            # adjust so that summing for epoch loss gives real MSE
            li = (loss*(img.shape[0]/len(dataloader.dataset))).detach()
            epoch_loss = epoch_loss + li
            iter_losses.append(li.item())
            #itbar.set_postfix(minibatch_loss=loss.item())
            pose_optimizer.step()
        with torch.no_grad():
            if world_size > 1:
                all_reduce(epoch_loss)
                all_reduce(I.grad)
                I.grad = I.grad/(len(dataloader)*world_size)
        image_optimizer.step()
        losses.append(epoch_loss.item())
        if test_every > 0 \
                and test_dataset is not None \
                and epoch % test_every == 0:
            # compute test loss
            with torch.no_grad():
                test_loss = 0
                for it, (ix, img) in enumerate(test_dataloader):
                    img = img.to(device)
                    A, T = affine_net(img.view(img.shape[0],-1))
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
                    # adjust so that summing for epoch loss gives real MSE
                    li = (loss*(img.shape[0]/len(test_dataloader.dataset))).detach()
                    test_loss += li
                all_reduce(test_loss)
                test_losses.append(test_loss.item())
        if rank == 0:
            epbar.set_postfix(epoch_loss=epoch_loss.item(),
                    test_loss=test_loss.item())
    return I.detach(), affine_net, losses, full_losses, iter_losses, test_losses


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


class MomentumPredictor(nn.Module):
    def __init__(self,
                 img_size=(1,1,256,256,256),
                 conv_layers=[(5,4,2),
                              (5,4,2),
                              (5,2,2),
                              (3,1,1)],
                 mlp_hidden=[256,128,64]):
        super(MomentumPredictor, self).__init__()
        self.img_size = img_size
        self.down_layers = conv_down_from_spec(conv_layers, img_size)
        from itertools import chain
        self.down_layers_params = nn.ParameterList(chain(*[p.parameters() \
                                             for p in self.down_layers]))
        Itest = torch.zeros(img_size, dtype=torch.float32)
        Itest,_,_ = self.down_net(Itest)
        n_features = Itest.view(1,-1).shape[1]
        del Itest
        #print(f"n_features={n_features}")
        self.mlp = MLP([n_features] + mlp_hidden)# + [n_features])

        #self.up_layers = conv_up_from_spec(conv_layers, img_size, 3)
        #self.up_layers_params = nn.ParameterList(chain(*[p.parameters() \
                                             #for p in self.up_layers]))
        last_layer_scale=0e-5
        self.dense_up = nn.Linear(mlp_hidden[-1], np.prod(img_size)*3)
        with torch.no_grad():
            self.dense_up.weight.mul_(last_layer_scale)
            self.dense_up.bias.mul_(last_layer_scale)
            #self.up_layers[-1].weight.mul_(last_layer_scale)
            #self.up_layers[-1].bias.mul_(last_layer_scale)
    def down_net(self, x):
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
        return d, inds, szs
    def up_net(self, d, inds, szs):
        for l, ix, sz in zip(self.up_layers, reversed(inds), reversed(szs)):
            if isinstance(l, nn.MaxUnpool3d):
                d = l(d, ix, output_size=sz)
            elif isinstance(l, nn.ConvTranspose3d):
                d = l(d, output_size=sz)
            else:
                d = l(d)
        return d
    def forward(self, x):
        d, inds, szs = self.down_net(x)
        sh = d.shape
        d = self.mlp(d.view(x.shape[0],-1))#.view(*sh)
        d = self.dense_up(d).view(x.shape[0],3,*self.img_size[2:])
        #d = self.up_net(d, inds, szs)
        return d

    
def deep_lddmm_atlas(dataset,
        I0=None,
        fluid_params=[1e-1,0.,.01],
        num_epochs=500,
        batch_size=2,
        reg_weight=.001,
        closed_form_image=False,
        image_update_freq=10, # how many iters between image updates
        momentum_net=None,
        momentum_preconditioning=True,
        lddmm_integration_steps=5,
        learning_rate_pose=1e-5,
        learning_rate_image=1e6,
        resume_checkpoint=False,
        checkpoint_every=0,
        checkpoint_pattern='checkpoints/{epoch}.pth',
        gpu=None,
        world_size=1,
        rank=0):
    from torch.utils.data import DataLoader, TensorDataset
    if gpu is None:
        device = 'cpu'
    else:
        device = f'cuda:{gpu}'
    if world_size > 1:
        sampler = DistributedSampler(dataset, 
                num_replicas=world_size,
                rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
            num_workers=8, pin_memory=True, shuffle=False)
    #I = I.clone()
    if I0 is None:
        # initialize base image to mean
        I0 = batch_average(dataloader, dim=0)
    I0 = I0.view(1,1,*I0.squeeze().shape)
    #I = DenseInterp(I0)
    #if gpu is not None:
        #I = DistributedDataParallel(I, device_ids=[gpu], output_device=gpu)
        #I = I.to(f'cuda:{gpu}')
    #else:
    epoch_losses = []
    iter_losses = []
    if momentum_net is None:
        momentum_net = MomentumPredictor(img_size=I0.shape)
    momentum_net = momentum_net.to(device)
    print(f"Momentum network has {sum([p.numel() for p in momentum_net.parameters()])} parameters")
    if world_size > 1:
        momentum_net = DistributedDataParallel(momentum_net,
                device_ids=[gpu], output_device=gpu)
    start_epoch = 0
    if resume_checkpoint:
        while True:
            cpfile = checkpoint_pattern.format(epoch=start_epoch)
            if not os.is_file(cpfile):
                if start_epoch > 0: # load previous state
                    cpfile = checkpoint_pattern.format(epoch=start_epoch-1)
                    I, sd = torch.load(cpfile, map_location=device)
                    momentum_net.load_state_dict(sd)
                break
            start_epoch += 1
    I = I0.to(device).detach()
    from torch.nn.functional import mse_loss
    pose_optimizer = torch.optim.Adam(momentum_net.parameters(),
                # below we roughly compensate for scaling the loss
                # the goal is to have learning rates that are independent of
                # number and size of minibatches, but it's tricky to accomplish
                                      lr=learning_rate_pose*len(dataloader),
                                      weight_decay=1e-2)
    image_optimizer = torch.optim.SGD([I],
                                      lr=learning_rate_image,
                                      weight_decay=0)
    metric = lm.FluidMetric(fluid_params)
    epbar = range(start_epoch, num_epochs)
    if rank == 0:
        epbar = tqdm(epbar, desc='epoch')
    for epoch in epbar:
        epoch_loss = 0.0
        itbar = dataloader
        if epoch > 1: # start using gradients for image after one epoch
            closed_form_image = False
        if closed_form_image:
            splatI = torch.zeros_like(I)
            splatw = torch.zeros_like(I)
            splatI.requires_grad_(False)
            splatw.requires_grad_(False)
        if not closed_form_image and I.grad is not None:
            image_optimizer.zero_grad()
        I.requires_grad_(True)
        for it, (ix, img) in enumerate(itbar):
            pose_optimizer.zero_grad()
            img = img.detach().to(I.device)
            m = momentum_net(img)
            if momentum_preconditioning:
                m.register_hook(metric.flat)
            h = lm.expmap(metric, m, num_steps=lddmm_integration_steps)
            Idef = lm.interp(I, h)
            v = metric.sharp(m)
            reg_term = 0
            if reg_weight > 0:
                reg_term = (v*m).sum()
            loss = (mse_loss(Idef, img, reduction='sum') + reg_term) \
                    / (img.numel())
            loss.backward()
            li = (loss*(img.shape[0]/len(dataloader.dataset))).detach()
            epoch_loss += li
            iter_losses.append(li.item())
            iter_losses.append(loss.item())
            #itbar.set_postfix(minibatch_loss=loss.item())
            pose_optimizer.step()
            del loss, reg_term, v, m, h, Idef, img
        with torch.no_grad():
            if world_size > 1:
                all_reduce(epoch_loss)
                all_reduce(I.grad)
                I.grad = I.grad/world_size
            # average over iterations
            I.grad = I.grad / len(dataloader)
        image_optimizer.step()
        epoch_losses.append(epoch_loss.item())
        if rank == 0:
            epbar.set_postfix(epoch_loss=epoch_loss)
    return I.detach(), momentum_net, epoch_losses, iter_losses


