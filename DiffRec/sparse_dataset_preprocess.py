import numpy as np
import pandas as pd
import pickle

with open('./datasets/amazon-datasets/Video_Games.pkl', 'rb') as f:
    data = pickle.load(f)

data = pd.DataFrame(data[0])
print(data)
data = data.sort_values(['reviewerID', 'unixReviewTime'])
data = data.drop(columns = ['unixReviewTime'])
data = np.array(data)



userids = {}
ruserids = {}
movieids = {}
rmovieids = {}
ucount = 0
mcount = 0
count = 0
newdata = [] # there are some duplicates so sizes change
for uid, mid in data:
    if uid not in userids:
        userids[uid] = ucount
        ucount+=1
        data[count,0] = userids[uid]
    else:
        data[count,0] = userids[uid]

    if mid not in movieids:
        movieids[mid] = mcount
        mcount+=1
        data[count,1] = movieids[mid]
    else:
        data[count,1] = movieids[mid]
    count += 1


current_user = 0
temp_list = []
train_list = []
valid_list = []
for user, movie in data:
    if current_user != user:
        for item in temp_list[0:int(len(temp_list)*.5)]:
            train_list.append(item)
        for item in temp_list[int(len(temp_list)*.5)+1:]:
            valid_list.append(item)
        temp_list = []
        current_user = user
    else:
        temp_list.append([user, movie])

print(np.array(train_list[0:50]))
print(np.array(valid_list[0:50]))

np.save('./datasets/amazon-toys_sparse/train_list.npy', np.array(train_list), allow_pickle=True)
np.save('./datasets/amazon-toys_sparse/valid_list.npy', np.array(valid_list), allow_pickle=True)
np.save('./datasets/amazon-toys_sparse/test_list.npy', np.array(valid_list), allow_pickle=True)