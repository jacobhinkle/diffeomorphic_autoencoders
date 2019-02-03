import torch
from torch.distributed import barrier
import os

from deepatlas import *
from cmdline import *

avgfile = f'{prefix}oasisavg_{suffix}.pth'
Iavg = torch.load(avgfile, map_location=loc)

reg_weightA=1e1
reg_weightT=1e1
dropout=.2
batch_size=8

deepaffinefile = f'{prefix}deepaffine_{suffix}.pth'
if not os.path.isfile(deepaffinefile): # compute deep affine atlas
    print("Deep affine atlas building")
    res = \
        deep_affine_atlas(oasis_ds,
                          test_dataset=oasis_test_ds,
                          test_every=5,
                          I=Iavg.clone(),
                          learning_rate_pose=5e-5,
                          learning_rate_image=2e4,
                          reg_weightA=reg_weightA,
                          reg_weightT=reg_weightT,
                          dropout=dropout,
                          num_epochs=100,
                          batch_size=batch_size,
                          gpu=gpu,
                          world_size=args.world_size,
                          rank=rank)
    if rank == 0: torch.save(res, deepaffinefile)
else:
    res = torch.load(deepaffinefile, map_location=loc)
I_deepaffine, affine_net, epoch_losses_deepaffine, full_losses_deepaffine, \
                iter_losses_deepaffine, test_losses_deepaffine = res
barrier()

if False:
    deepaffinetestfile = f'{prefix}deepaffine_test_{suffix}.pth'
    if not os.path.isfile(deepaffinetestfile):
        print("Deep affine atlas building TEST")
        _, _, deepaffine_test_loss, _, _, _ = \
            deep_affine_atlas(oasis_test_ds,
                              I=I_deepaffine.to(loc),
                              affine_net=affine_net.to(loc),
                              learning_rate_pose=0e-5,
                              learning_rate_image=0e4,
                              reg_weightA=reg_weightA,
                              reg_weightT=reg_weightT,
                              dropout=dropout,
                              num_epochs=1,
                              batch_size=batch_size,
                              gpu=gpu,
                              world_size=args.world_size,
                              rank=rank)
        print(f"Test loss: {deepaffine_test_loss}")
        if rank == 0:
            torch.save((deepaffine_test_loss,), deepaffinetestfile)
    barrier()

if rank == 0:
    for ds, stdfile in [(oasis_ds, f'{prefix}deepaffinestd_{suffix}.h5'),
            (oasis_test_ds, f'{prefix}deepaffinestd_test_{suffix}.h5')]:
        if not os.path.isfile(stdfile): # standardize data
            print(f"Creating standardized dataset in file {stdfile}")
            affine_net = affine_net.to(loc)
            eye = torch.eye(3).view(1,3,3).type(Iavg.dtype).to(loc)
            with h5py.File(stdfile, 'w') as f:
                f.create_dataset('atlas', data=I_deepaffine.cpu().numpy())
                d = f.create_dataset('skullstripped',
                        (len(ds), *I_deepaffine.shape[2:]),
                        dtype=np.float32,
                        compression='lzf',
                        chunks=(1, *I_deepaffine.shape[2:]))
                with torch.no_grad():
                    for ii in tqdm(range(len(ds))):
                        i, J = ds[ii]
                        J = J.unsqueeze(1).to(loc)
                        A, T = affine_net(J)
                        Ainv, Tinv = lm.affine_inverse(A+eye, T)
                        Jdef = lm.affine_interp(J, Ainv, Tinv).cpu()
                        d[i,...] = Jdef.numpy()
