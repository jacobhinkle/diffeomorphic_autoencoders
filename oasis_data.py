import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
from utils import tqdm
import numpy as np


class OASISDataset(Dataset):
    def __init__(self,
                 h5path='data/oasis3.h5',
                 crop=None,
                 first=None,
                 pooling=None,
                 one_scan_per_subject=False # take only one image per subject
                 ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.h5path = h5path
        self.crop = crop
        self.first = first
        self.pooling = pooling
        self.one_scan_per_subject = one_scan_per_subject
        if pooling is not None:
            raise Exception("Pooling inside the OASIS dataloader is deprecated.  See OASISDownscaling.ipynb for examples of how to pre-pool the data.")
        # build list of indices to use
        if one_scan_per_subject:
            import pandas as pd
            meta = pd.read_hdf(h5path, key='metadata').reset_index()
            firstinds = meta.groupby('Subject').apply(lambda x: x.first_valid_index())
            ids = firstinds.tolist()
        else:
            with h5py.File(self.h5path, 'r') as f:
                ids = range(f['skullstripped'].shape[0])
        if first is not None:
            ids = ids[:first]
        self.scan_ids = list(ids)

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, idx):
        # first lookup actual index in ids map
        if self.scan_ids is not None:
            idint = self.scan_ids[idx]
        with h5py.File(self.h5path, 'r') as f:
            ss = f['skullstripped']
            if self.crop is None:
                sl = ss[idint,...]
            else:
                sl = ss[idint,
                                        self.crop[0,0]:self.crop[0,1],
                                        self.crop[1,0]:self.crop[1,1],
                                        self.crop[2,0]:self.crop[2,1]
                                       ]
            I = torch.as_tensor(sl).type(torch.float32).unsqueeze(0).contiguous()
        if self.pooling is not None:
            I = F.avg_pool3d(I, self.pooling)
        return idx, I

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

docrop = 1
if docrop:
    crop = np.asarray(
    [[ 47, 212],
     [ 16, 251],
     [  9, 228]])
    if True:
        crop[:,0] -= 1
        crop[:,1] += 1
else:
    crop = None
ds_first = 2000
one_scan_per_subject = True
ds_pooling = None
oasis_ds = OASISDataset(crop=crop,
                        first=ds_first,
                        pooling=ds_pooling,
                        one_scan_per_subject=one_scan_per_subject)
