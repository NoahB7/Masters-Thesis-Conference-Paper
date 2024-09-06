import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
from torch.utils.data import Dataset

# takes in tuples of data points, (userid, movieid) no rating present only 4's and 5's represented as positive reviews
def data_load(train_path, valid_path, test_path):

    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)
    print(len(train_list))
    # load in raw data

    # print(train_list.shape, valid_list.shape, test_list.shape)

    # train_list = train_list[:int(train_list.shape[0]*.2)]
    # valid_list = valid_list[:int(valid_list.shape[0]*.2)]
    # test_list = test_list[:int(test_list.shape[0]*.2)]

    uid_max = 0
    iid_max = 0
    train_dict = {}

    # count number of unique users and items using dicts
    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    # add one to get count since ids start at 0
    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    # original data is in shape  (429993 , 2) for 429,993 reviews of a user a on movie b

    # np.ones_like is just all 1's (a place holder I think, not sure why it has to be done this way)

    # the operations below transform the 429k reviews into the sparse matrix format of user 
    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))
                    #  shape=(n_user, n_item))
    
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
        
    
    return train_data, valid_y_data, test_y_data, n_user, n_item

# takes in data of the format (userid, movieid, rating) noisier training with ratings present rather than just binary classifications
def custom_data_load(train_path, valid_path, test_path):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    train_dict = {}

    # count number of unique users and items using dicts
    for uid, iid, r in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    # add one to get count since ids start at 0
    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')

    train_data = np.zeros((n_user,n_item))
    for pair in train_list:
        train_data[pair[0], pair[1]] = pair[2]

    valid_y_data = np.zeros((n_user,n_item))
    for pair in valid_list:
        valid_y_data[pair[0], pair[1]] = pair[2]
        
    test_y_data = np.zeros((n_user,n_item))
    for pair in test_list:
        test_y_data[pair[0], pair[1]] = pair[2]

    print(train_data.shape)
    print(valid_y_data.shape)
    print(test_y_data.shape)
    return train_data, valid_y_data, test_y_data, n_user, n_item


class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)
