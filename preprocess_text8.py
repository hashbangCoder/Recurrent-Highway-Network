# Original code from https://github.com/julian121266/RecurrentHighwayNetworks/blob/master/data/create_text8.py

from __future__ import division, print_function, unicode_literals
import numpy as np
import zipfile
import h5py
import os


def convert_to_batches(serial_data, length, bs):
    assert serial_data.size % length == 0
    num_sequences = serial_data.size // length
    assert num_sequences % bs == 0
    num_batches = num_sequences // bs
    serial_data = serial_data.reshape((bs, num_batches * length))
    serial_data = np.vstack(np.hsplit(serial_data, num_batches)).T[:, :, None]
    return serial_data

batch_size = 128
# Batch size which will be used for training.
# Needed to maintain continuity of data across batches.
seq_len = 50
# Number of characters in each sub-sequence.
# Limits the number of time-steps that the gradient is back-propagated.
num_test_chars = 5000000
# Number of characters which will be used for testing.
# An equal number of characters will be used for validation.

with open('data/Datasets/text8','r') as f:
    raw_data = f.read()

print("Preparing data...")
raw_data = np.fromstring(raw_data, dtype=np.uint8)
unique, data = np.unique(raw_data, return_inverse=True)

print("Vocabulary size:", unique.shape)
train_data = data[: -2 * num_test_chars]
valid_data = data[-2 * num_test_chars: -num_test_chars]
test_data = data[-num_test_chars:]

print("Done.")

print("Creating Text8 character-level HDF5 dataset ...")
bs_data_dir = os.environ.get('data/', '.')
hdf_file = os.path.join(bs_data_dir, 'Text8_Torch.hdf5')
f = h5py.File(hdf_file, 'w')
description = """
The Text8 Wikipedia dataset, prepared for character-level language
modeling.

The data was obtained from the link:
http://mattmahoney.net/dc/text8.zip

Variants
========

split: Split into 'training', 'validation' and 'test' tests of size 90, 5 and
5 million characters respectively. Each sequence is {} characters long. The
dataset has been prepared expecting minibatches of {} sequences.
""".format(seq_len, batch_size)
f.attrs['description'] = description
f.attrs['unique'] = unique

variant = f.create_group('split')
group = variant.create_group('training')
group.create_dataset(name='default', data=train_data, compression='gzip')

group = variant.create_group('validation')
group.create_dataset(name='default', data=valid_data, compression='gzip')

group = variant.create_group('test')
group.create_dataset(name='default', data=test_data, compression='gzip')

f.close()
print("Done.")