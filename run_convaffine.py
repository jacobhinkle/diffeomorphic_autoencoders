import torch
from torch.distributed import barrier
import os

from cmdline import *
from atlas import *

avgfile = f'{prefix}oasisavg_{suffix}.pth'
Iavg = torch.load(avgfile, map_location=loc)

reg_weightA=1e1
reg_weightT=1e1

convaffinefile = f'{prefix}convaffine_{suffix}.pth'
if not os.path.isfile(convaffinefile): # compute affine atlas
    print("Conventional affine atlas building")
    As = torch.zeros((len(oasis_ds),3,3), dtype=torch.float32).to(loc)
    Ts = torch.zeros((len(oasis_ds),3), dtype=torch.float32).to(loc)
    res = affine_atlas(
        dataset=oasis_ds,
        I=Iavg, As=As, Ts=Ts,
        affine_steps=1,
        num_epochs=250,
        learning_rate_A=1e-4,
        learning_rate_T=1e-3,
        learning_rate_I=1e3,
        reg_weightA=reg_weightA,
        reg_weightT=reg_weightT,
        batch_size=8,
        gpu=gpu,
        world_size=args.world_size,
        rank=rank)
    # save result
    if rank == 0: torch.save(res, convaffinefile)
else:
    res = torch.load(convaffinefile, map_location='cpu')
I_affine, As_train, Ts_train, _, _ = res
barrier()

# On the test set, use same atlas-building code but with zero learning rate for
# the image
convaffinetestfile = f'{prefix}convaffine_test_{suffix}.pth'
if not os.path.isfile(convaffinetestfile): # conventional lddmm atlas
    print("Conventional Affine Test")
    As_test = torch.zeros((len(oasis_test_ds),3,3), dtype=torch.float32).to('cpu')
    Ts_test = torch.zeros((len(oasis_test_ds),3), dtype=torch.float32).to('cpu')
    res = affine_atlas(
        dataset=oasis_test_ds,
        I=I_affine, As=As_test, Ts=Ts_test,
        affine_steps=250,
        num_epochs=1,
        learning_rate_A=1e-4,
        learning_rate_T=1e-3,
        learning_rate_I=0e5,
        reg_weightA=reg_weightA,
        reg_weightT=reg_weightT,
        batch_size=8,
        gpu=gpu,
        world_size=args.world_size,
        rank=rank)
    if rank == 0: torch.save(res, convaffinetestfile)
else:
    res = torch.load(convaffinetestfile, map_location='cpu')
barrier()
_, As_test, Ts_test, _, _ = res

if rank == 0:
    for ds, stdfile, As, Ts in [(oasis_ds, f'{prefix}convaffinestd_{suffix}.h5', As_train, Ts_train),
            (oasis_test_ds, f'{prefix}convaffinestd_test_{suffix}.h5', As_test, Ts_test)]:
        As = As.to(loc)
        Ts = Ts.to(loc)
        if not os.path.isfile(stdfile): # standardize data
            print(f"Creating standardized dataset in file {stdfile}")
            eye = torch.eye(3).view(1,3,3).type(Iavg.dtype).to(loc)
            with h5py.File(stdfile, 'w') as f:
                f.create_dataset('atlas', data=I_affine.cpu().numpy())
                d = f.create_dataset('skullstripped',
                        (len(ds), *I_affine.shape[2:]),
                        dtype=np.float32,
                        compression='lzf',
                        chunks=(1, *I_affine.shape[2:]))
                with torch.no_grad():
                    for ii, (i, J) in tqdm(enumerate(ds)):
                        i, J = ds[ii]
                        J = J.unsqueeze(1).to(loc)
                        A, T = As[[i],...].to(loc), Ts[[i],...].to(loc)
                        Ainv, Tinv = lm.affine_inverse(A+eye, T)
                        Jdef = lm.affine_interp(J, Ainv, Tinv).cpu()
                        d[i,...] = Jdef.numpy()
