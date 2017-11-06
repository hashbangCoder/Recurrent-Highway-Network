# Original Code from https://github.com/julian121266/RecurrentHighwayNetworks/blob/master/data/reader.py

# This file is adapted from the tool provided with Tensorflow for
# reading the Penn Treebank dataset. The original copyright notice is
# provided below.
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for training on the Hutter Prize dataset."""
from __future__ import absolute_import
import os
import numpy as np
import h5py
import math
from tqdm import tqdm


class Dictionary():
    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.vocab_size = 0

    def add_token(self, token):
        if token not in self.token2id:
            self.token2id[token] = self.vocab_size + 1
            self.id2token[self.vocab_size + 1] = token
            self.vocab_size += 1


def text8_token_mapping(mapping):
    # token_map is from given tokens to ascii values
    # reverse_map is from ascii to given tokens
    token_map = {}
    reverse_map = {}
    for ind, val in np.ndenumerate(mapping):
        token_map[ind] = val
        reverse_map[val] = ind
    return token_map, reverse_map


class Text8DataLoader:
    def __init__(self, data_path, max_epochs):
        assert os.path.isfile(data_path), 'Data file doesnt exist at path'
        with open(data_path, 'r'):
            dataset = h5py.File(data_path, 'r')
        self.train_data = dataset['split']['training']['default'].value
        self.test_data = dataset['split']['test']['default'].value
        self.train_size = self.train_data.shape[0]
        self.test_size = self.test_data.shape[0]
        self.mapping, self.rev_mapping = text8_token_mapping(dataset.attrs['unique'])
        self.map_func = np.vectorize(lambda x: self.mapping[x])
        self.rev_map_func = np.vectorize(lambda x: self.rev_mapping[x])

        self._iter = 0
        self._test_iter = 0
        self.epoch = 0
        self.step_counter = 0
        self.max_epochs = max_epochs
        self.end_flag = False
        self.pbar = tqdm(total=self.train_size)

    def get_batch(self, batch_size, seq_length):
        if self.epoch > self.max_epochs:
            self.end_flag = True
            return None

        # if cannot accomodate batch_size, find smaller batch_size
        if self._iter + (seq_length * batch_size) - 1 >= self.train_size:
            batch_size = math.floor((self.train_size - self._iter - 1) / seq_length)
            self._iter = 0
            self.epoch += 1

        iter_size = batch_size * seq_length
        batch = self.map_func(self.train_data[self._iter: self._iter + iter_size])
        labels = self.map_func(self.train_data[self._iter + 1: self._iter + iter_size])

        self._iter += iter_size
        self.pbar.update(iter_size)
        self.step_counter += 1
        return batch.reshape(shape=(batch_size, seq_length)), \
               labels.reshape(shape=(batch_size, seq_length))

    def get_eval_batch(self, batch_size, seq_length):
        # End of split/epoch
        if self._test_iter == -1:
            return None

        if self._test_iter + (seq_length * batch_size) - 1 >= self.train_size:
            batch_size = math.floor((self.train_size - self._test_iter - 1) / seq_length)
            self._test_iter = -1

        iter_size = batch_size * seq_length
        batch = self.map_func(self.train_data[self._test_iter: self._test_iter + iter_size])
        labels = self.map_func(self.train_data[self._test_iter + 1: self._test_iter + iter_size])

        return batch.reshape(shape=(batch_size, seq_length)),\
               labels.reshape(shape=(batch_size, seq_length))



# def enwik8_raw_data(data_path=None, num_test_symbols=5000000):
#     """Load raw data from data directory "data_path".
#
#     The raw Hutter prize data is at:
#     http://mattmahoney.net/dc/enwik8.zip
#
#     Args:
#         data_path: string path to the directory where simple-examples.tgz has
#             been extracted.
#         num_test_symbols: number of symbols at the end that make up the test set
#
#     Returns:
#         tuple (train_data, valid_data, test_data, unique)
#         where each of the data objects can be passed to hutter_iterator.
#     """
#
#     data_path = os.path.join(data_path, "enwik8")
#     if os.path.isfile(data_path):
#         with open(data_path, 'r') as f:
#             raw_data = f.read()
#             raw_data = np.fromstring(raw_data, dtype=np.uint8)
#             unique, data = np.unique(raw_data, return_inverse=True)
#             train_data = data[: -2 * num_test_symbols]
#             valid_data = data[-2 * num_test_symbols: -num_test_symbols]
#             test_data = data[-num_test_symbols:]
#     else:
#         raise Exception('Cannot locate wiki8 dataset.')
#     return train_data, valid_data, test_data, unique
#
#
# def text8_raw_data(data_path=None, num_test_symbols=5000000):
#     """Load raw data from data directory "data_path".
#
#     The raw text8 data is at:
#     http://mattmahoney.net/dc/text8.zip
#
#     Args:
#         data_path: string path to the directory where simple-examples.tgz has
#             been extracted.
#         num_test_symbols: number of symbols at the end that make up the test set
#
#     Returns:
#         tuple (train_data, valid_data, test_data, unique)
#         where each of the data objects can be passed to text8_iterator.
#     """
#
#     data_path = os.path.join(data_path, "text8")
#     if os.path.isfile(data_path):
#         with open(data_path, 'r') as f:
#             raw_data = f.read(data_path)
#             raw_data = np.fromstring(raw_data, dtype=np.uint8)
#             unique, data = np.unique(raw_data, return_inverse=True)
#             train_data = data[: -2 * num_test_symbols]
#             valid_data = data[-2 * num_test_symbols: -num_test_symbols]
#             test_data = data[-num_test_symbols:]
#     else:
#             raise Exception('Cannot locate wiki8 dataset.')
#
#     return train_data, valid_data, test_data, unique
#
#
# def data_iterator(raw_data, batch_size, num_steps):
#     """Iterate on the raw Hutter prize data.
#
#     This generates batch_size pointers into the raw Hutter Prize data, and allows
#     minibatch iteration along these pointers.
#
#     Args:
#         raw_data: one of the raw data outputs from ptb_raw_data.
#         batch_size: int, the batch size.
#         num_steps: int, the number of unrolls.
#
#     Yields:
#         Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
#         The second element of the tuple is the same data time-shifted to the
#         right by one.
#
#     Raises:
#         ValueError: if batch_size or num_steps are too high.
#     """
#     raw_data = np.array(raw_data, dtype=np.int32)
#
#     data_len = len(raw_data)
#     batch_len = data_len // batch_size
#     data = np.zeros([batch_size, batch_len], dtype=np.int32)
#     for i in range(batch_size):
#         data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
#
#     epoch_size = (batch_len - 1) // num_steps
#
#     if epoch_size == 0:
#         raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
#
#     for i in range(epoch_size):
#         x = data[:, i*num_steps:(i+1)*num_steps]
#         y = data[:, i*num_steps+1:(i+1)*num_steps+1]
#         yield (x, y)