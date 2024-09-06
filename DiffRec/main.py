"""
Train a diffusion model for recommendation
"""
from sklearn.metrics import ndcg_score

import argparse
from ast import parse
import os
import time
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models.gaussian_diffusion as gd
from models.DNN import DNN
from models.CustomDNN import CustomDNN
import evaluate_utils
import data_utils
from copy import deepcopy

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
# def worker_init_fn(worker_id):
#     np.random.seed(random_seed + worker_id)
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='amazon-toys_sparse_original', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='./datasets/amazon-toys_sparse/', help='load data path') # CLEAN DATA IS ONLY MOVIES THE USER RATED A 4 or 5, NOISY CONTAINS UNDER 4
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate') # 0.0001 for others 0.00005 for amazon book
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[1, 5, 10, 20]')
parser.add_argument('--tst_w_val', default=False, help='test with validation')
parser.add_argument('--cuda', default=True, help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

# params for the model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=2000, help='embedding size')
parser.add_argument('--p_uncond', type=float, default=0.2, help='probability when using classifier-free guidance to generate without guidance')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=100, help='diffusion steps') # 100 originally, 200
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating') # 0.02 originally, 0.2
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference') # investigate
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')


args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


### DATA LOAD ###
train_path = args.data_path + 'train_list.npy'
valid_path = args.data_path + 'valid_list.npy'
test_path = args.data_path + 'test_list.npy'

# train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.custom_data_load(train_path, valid_path, test_path)
train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
print("made it")
train_data = train_data[:int(train_data.shape[0]*.05)]
valid_y_data = valid_y_data[:int(valid_y_data.shape[0]*.05)]

train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
# train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)
valid_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

print('data ready.')


### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device, args.p_uncond).to(device)

## Build MLP ###
out_dims = eval(args.dims) + [n_item]
in_dims = out_dims[::-1]
model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)
# model = CustomDNN(n_item, args.emb_size, time_type="cat", norm=args.norm).to(device)



optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("models ready.")

param_num = 0
mlp_num = sum([param.nelement() for param in model.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = mlp_num + diff_num
print("Number of all parameters:", param_num)


def custom_evaluate(valid_loader, valid_y_data, train_data, topN):
    model.eval()
    target_items = []
    target_indices = []
    predict_items = []
    predict_indices = []
    for row in valid_y_data:
        target_indices.append(row.nonzero()[0].tolist())
        target_items.append(row[row != 0])
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            known_train_data = train_data[batch_idx*args.batch_size:batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]
            batch = batch.to(device)
            prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise)
            prediction[known_train_data.nonzero()] = -np.inf

            values, indices = torch.topk(prediction, 20) 
            values = values.cpu().numpy().tolist()
            indices = indices.cpu().numpy().tolist()
            print(values[0])
            predict_items.extend(values*5)
            predict_indices.extend(indices)

    test_results = evaluate_utils.custom_computeTopNAccuracy(target_indices, predict_items, predict_indices, topN)

    return test_results


# test_loader(train_data technically), valid_y_data, train_data
# MAP metric for all @ 1 - N
# nDCG for using actual user reviews not just 4 and 5 equal to good and below is bad
def evaluate(data_loader, data_te, mask_his, topN):
    model.eval()
    e_idxlist = list(range(mask_his.shape[0])) # creates a list of user ids, since shape[0] is number of users
    e_N = mask_his.shape[0]  # total count of users

    predict_items = []
    target_items = []
    # get indices for all movies all users rated a 5
    for i in range(e_N):
        # .nonzero() returns two matrices one for each indice of a list, array1, array2, array1 is row indices array2 is columns, since were just working with a 1d array for 
        # each user we just want the columns which is what [1] is for, taking the second array only
        target_items.append(data_te[i, :].nonzero()[1].tolist()) # creates a row of indices for each user where they rated a 5
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]] # slicing to match which batch of data we are working with
            batch = batch.to(device)
            prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise) # p_sample includes q_sample (noising) and denoising prediction all in one method
            prediction[his_data.nonzero()] = -np.inf # setting predictions to -infinity where we already know users rated a 5 or 4

            _, indices = torch.topk(prediction, 100) # getting the 100 largest values from prediction
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results

best_nDCG, best_epoch = -100, 0
best_test_result = None
prev = ""
print("Start training...")
for epoch in range(1, args.epochs+1):

    model.train()
    model.training = True
    start_time = time.time()

    batch_count = 0
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        batch_count += 1
        optimizer.zero_grad()
        losses = diffusion.training_losses(model, batch, args.reweight)
        loss = losses["loss"].mean()
        total_loss += loss
        loss.backward()
        optimizer.step()
    
    if epoch % 5 == 0:
        model.training = False
        # valid_results = custom_evaluate(valid_loader, valid_y_data, train_data, eval(args.topN))
        valid_results = evaluate(valid_loader, valid_y_data, train_data, eval(args.topN))
        evaluate_utils.print_results(None, valid_results, None)
        
        if valid_results[2][2] > best_nDCG: # nDCG @ 10
            best_nDCG, best_epoch = valid_results[2][2], epoch
            best_results = valid_results

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)

            if prev:
                os.remove(prev)
            
            torch.save(model, '{}{}_{}.pth'.format(args.save_path, args.dataset, valid_results))
            prev = '{}{}_{}.pth'.format(args.save_path, args.dataset, valid_results)
            
    
    print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
evaluate_utils.print_results(None, best_results, None)
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


# the data was way too big initially
# movie lens is kind of simplistic just a starting point
# the evaluation takers forever, modifying code so that it doesnt evaluate on the entire dataset (who thought that was a good idea)
# move on to a different dataset that isnt just 1s and 0s (I think) I havent actually looked at the other datasets yet so I dont know what the data looks like
# make my modifications, compare?