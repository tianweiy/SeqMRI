import pathlib
import h5py

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

train_dir = pathlib.Path('/home/tianweiy/ActiveMRI-Release/datasets/brain/train')
val_dir = pathlib.Path('/home/tianweiy/ActiveMRI-Release/datasets/brain/val')

def remove_kspace(path):
    for fname in tqdm(list(path.iterdir())):
        if not fname.name.endswith('.h5'):
            continue  # Skip directories
        new_dir = fname.parent.parent / pathlib.Path(str(fname.parent.name) + '_no_kspace')
        if not new_dir.exists():
            new_dir.mkdir(parents=False)
        new_filename = new_dir / fname.name
        if new_filename.exists():
            continue  # Skip already done files
        f = h5py.File(fname, 'r')
        fn = h5py.File(new_filename, 'w')
        for at in f.attrs:
            fn.attrs[at] = f.attrs[at]
        for dat in f:
            if dat == 'kspace':
                continue
            f.copy(dat, fn)

# Run the calls below to remove the stored kspace from multicoil_ brain .h5 file, which will save on I/O later.
# We don't need the multicoil kspace since we will construct singlecoil kspace from the ground truth images.
# Commented out for safety.

remove_kspace(train_dir)
remove_kspace(val_dir)

def create_train_test_split(orig_train_dir, target_test_dir, test_frac):
    """
    Creates a train and test split from the provided training data. Works by
    moving random volumes from the training directory to a new test directory.

    WARNING: Only use this function once to create the required datasets!
    """
    import shutil
    np.random.seed(0)
    
    files = sorted(list(orig_train_dir.iterdir()))
    target_test_dir.mkdir(parents=False, exist_ok=False)

    permutation = np.random.permutation(len(files))
    test_indices = permutation[:int(len(files) * test_frac)]
    test_files = list(np.array(files)[test_indices])

    for i, file in enumerate(test_files):
        print("Moving file {}/{}".format(i + 1, len(test_files)))
        shutil.move(file, target_test_dir / file.name)
        
        
def count_slices(data_dir, dataset):
    vol_count, slice_count = 0, 0
    for fname in data_dir.iterdir():
        with h5py.File(fname, 'r') as data:
            if dataset == 'brain':
                gt = data['reconstruction_rss'][()]
            vol_count += 1
            slice_count += gt.shape[0]
    print(f'{vol_count} volumes, {slice_count} slices')

# For both Knee and Brain data, split off 20% of train as test
dataset = 'brain'  # or 'brain'
train_dir = pathlib.Path('/home/tianweiy/ActiveMRI-Release/datasets/brain/train_no_kspace')
val_dir = pathlib.Path('/home/tianweiy/ActiveMRI-Release/datasets/brain/val_no_kspace')
test_dir = pathlib.Path('/home/tianweiy/ActiveMRI-Release/datasets/brain/test_no_kspace')

test_frac = 0.2

# Run this to split of test_frac of train data into test data.
# Commented out for safety.

# create_train_test_split(train_dir, test_dir, test_frac)

count_slices(train_dir, dataset)
count_slices(val_dir, dataset)
count_slices(test_dir, dataset)
