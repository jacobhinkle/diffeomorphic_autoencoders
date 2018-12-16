import nibabel as nib
import h5py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

fs_dir = '.'
skullstripped_dir = fs_dir + '/skullstripped'
h5_file = fs_dir + '/oasis3.h5'
imgs = []
allfiles = sorted(os.listdir(skullstripped_dir))
ssfiles = []
labels = []
for f in allfiles:
    label, ext = os.path.splitext(f)
    label = label[:-len('_skullstripped')]
    if ext != '.mgz': continue
    ssfiles.append(f)
    labels.append(label)
print(f"Found {len(ssfiles)} skullstripped .mgz files")
subject_meta = pd.read_csv(fs_dir + '/subject_metadata.csv',
        index_col='Subject', na_filter=False)
subjects = [l[:8] for l in labels]
sessions = [l[-5:] for l in labels]
scan_session = pd.DataFrame({'Subject':subjects, 'Session':sessions},
        index=labels)
scan_session.index.name = 'Freesurfer_label'
scan_meta = scan_session.join(subject_meta, on='Subject')
scan_meta.to_hdf(h5_file, key='metadata', mode='w')
with h5py.File(h5_file, 'a') as f:
    imds = f.create_dataset('skullstripped', shape=(len(ssfiles),256,256,256),
            dtype=np.uint8, compression="lzf", chunks=(1,256,256,256))
    for i, ssfile in enumerate(tqdm(ssfiles)):
        im = np.asarray(nib.load(os.path.join(skullstripped_dir,ssfile)).get_data())
        imds[i,...] = im
    f.create_dataset('scan_labels', data=np.asarray(labels, dtype='S'))
print(f"Successfully created {h5_file} with dataset named 'skullstripped'")

