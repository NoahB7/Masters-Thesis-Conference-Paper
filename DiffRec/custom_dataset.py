import numpy as np
import random
import pandas as pd
import scipy.sparse as sp
import pickle


# data = pd.read_csv('datasets/archive/Books_rating.csv')

# data = data.drop(columns=['Price','profileName','review/helpfulness','review/summary','review/text'])

# data = data[data['review/score'] >= 4.0]

# for col in data.columns:
#     bools = pd.isnull(data[col])
#     if bools.any():
#         print('nulls found in ', col)
#         print(len(data[bools]))

# titlebools = pd.notnull(data['Title'])
# idbools = pd.notnull(data['User_id'])

# bools = [a and b for a, b in zip(titlebools, idbools)]
# data = data[bools]

# print(data)


# ----------------------------------------------------------------------------------------------------------------------------------------------

data = pd.read_csv('datasets/ml-1m_custom/ratings.dat', names=['stuff'])

data[['UID','MID','R','T']] = data['stuff'].str.split('::', expand=True) # convert :: delimited data
data = data.drop(columns = ['stuff']) # remove non numeric data
for col in data.columns:
    data[col] = pd.to_numeric(data[col]) # convert all numeric data to integers

data = data[data['R'] >= 4] # remove ratings 3 and below

data = data.sort_values(['UID','T']) # sort by users and then time, last in the list being the latest
data = data.drop(columns = ['T']) # remove unecessary columns once filtering has been done
data = np.array(data)


# ----------------------------------------------------------------------------------------------------------------------------------------------

# custom_list = np.load('./datasets/ml-1m_custom/train_list.npy', allow_pickle=True)
# train_list = np.load('./datasets/ml-1m_clean/train_list.npy', allow_pickle=True)
# test_list = np.load('./datasets/ml-1m_clean/test_list.npy', allow_pickle=True)
# valid_list = np.load('./datasets/ml-1m_clean/valid_list.npy', allow_pickle=True)

# print(custom_list.shape)
# print(train_list.shape, test_list.shape, valid_list.shape)
# print(train_list[0:50], test_list[0:50], valid_list[0:50])

# print(train_list[0:50])
# print(train_list1[0:50])

# print(train_list[40:200])
# uid_max = 0
# iid_max = 0
# train_dict = {}
# for uid, iid in train_list:
#         if uid not in train_dict:
#             train_dict[uid] = []
#         train_dict[uid].append(iid)
#         if uid > uid_max:
#             uid_max = uid
#         if iid > iid_max:
#             iid_max = iid
    
# # add one to get count since ids start at 0
# n_user = uid_max + 1
# n_item = iid_max + 1
# print(f'user num: {n_user}')
# print(f'item num: {n_item}')

# # original data is in shape  (429993 , 2) for 429,993 reviews of a user a on movie b

# # np.ones_like is just all 1's (a place holder I think, not sure why it has to be done this way)

# # the operations below transform the 429k reviews into the sparse matrix format of user 
# train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
#                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
#                 shape=(n_user, n_item))
#                 #  shape=(n_user, n_item))

# print(train_data[0:50])