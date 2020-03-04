import json
import torch
from torch import nn
import os
import numpy as np
from uni_model_2 import *
# from data_loader import MusicArrayLoader
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split

from ptb_v2 import *


# some initialization
with open('uni_model_config_2.json') as f:
    args = json.load(f)
if not os.path.isdir('log'):
    os.mkdir('log')
if not os.path.isdir('params'):
    os.mkdir('params')
save_path = 'params/{}.pt'.format(args['name'])

from datetime import datetime
timestamp = str(datetime.now())
save_path_timing = 'params/{}.pt'.format(args['name'] + "_" + timestamp)

# model dimensions
EVENT_DIMS = 342
RHYTHM_DIMS = 3
NOTE_DIMS = 16
TEMPO_DIMS = 264
VELOCITY_DIMS = 126
CHROMA_DIMS = 24

class SimpleModel(nn.Module):
    def __init__(self, roll_dims, hidden_dims=32):
        super(SimpleModel, self).__init__()
        self.gru_r = nn.GRU(roll_dims, hidden_dims, batch_first=True, bidirectional=True)
        self.gru_n = nn.GRU(roll_dims, hidden_dims // 4, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dims * 2)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dims // 2)
        self.linear_r = nn.Linear(hidden_dims * 2, 1)
        self.linear_n = nn.Linear(hidden_dims // 2, 1)
    
    def forward(self, x):
        r_predict = self.gru_r(x)[-1]
        r_predict = self.bn1(torch.cat([r_predict[0], r_predict[1]], dim=-1))
        n_predict = self.gru_n(x)[-1]
        n_predict = self.bn2(torch.cat([n_predict[0], n_predict[1]], dim=-1))
        r_predict = self.linear_r(r_predict)
        n_predict = self.linear_n(n_predict)

        return r_predict, n_predict


model = SimpleModel(EVENT_DIMS)
optimizer = optim.Adam(model.parameters(), lr=args['lr'])

if torch.cuda.is_available():
    print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('CPU mode')

step, pre_epoch = 0, 0
batch_size = args["batch_size"]
model.train()

# dataloaders
is_shuffle = True
data_lst, rhythm_lst, note_density_lst, chroma_lst = get_classic_piano()
tlen, vlen = int(0.8 * len(data_lst)), int(0.9 * len(data_lst))
train_ds_dist = MusicAttrDataset2(data_lst, rhythm_lst, note_density_lst, 
                                chroma_lst, mode="train")
train_dl_dist = DataLoader(train_ds_dist, batch_size=batch_size, shuffle=is_shuffle, num_workers=0)
val_ds_dist = MusicAttrDataset2(data_lst, rhythm_lst, note_density_lst, 
                                chroma_lst, mode="val")
val_dl_dist = DataLoader(val_ds_dist, batch_size=batch_size, shuffle=is_shuffle, num_workers=0)
test_ds_dist = MusicAttrDataset2(data_lst, rhythm_lst, note_density_lst, 
                                chroma_lst, mode="test")
test_dl_dist = DataLoader(test_ds_dist, batch_size=batch_size, shuffle=is_shuffle, num_workers=0)
dl = train_dl_dist
print(len(train_ds_dist), len(val_ds_dist), len(test_ds_dist))


is_class = args["is_class"]
is_res = args["is_res"]
# end of initialization


def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N


def loss_function(r_predict, r_density_lst, n_predict, n_density_lst):
    criterion = torch.nn.L1Loss(reduction="mean")    
    return criterion(r_predict.squeeze(), r_density_lst), criterion(n_predict.squeeze(), n_density_lst)


def train(d_oh, r_density_lst, n_density_lst):
    
    optimizer.zero_grad()
    r_predict, n_predict = model(d_oh)
    
    # calculate loss
    CE_R, CE_N = loss_function(r_predict, r_density_lst,
                               n_predict, n_density_lst)

    loss = CE_R + CE_N
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    
    output = CE_R.item(), CE_N.item()
    return output


def evaluate(d_oh, r_density_lst, n_density_lst):
    
    r_predict, n_predict = model(d_oh)

    # calculate loss
    CE_R, CE_N = loss_function(r_predict, r_density_lst,
                               n_predict, n_density_lst)

    loss = CE_R + CE_N
    output = CE_R.item(), CE_N.item()
    return output


def convert_to_one_hot(input, dims):
    if len(input.shape) > 1:
        input_oh = torch.zeros((input.shape[0], input.shape[1], dims)).cuda()
        input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
    else:
        input_oh = torch.zeros((input.shape[0], dims)).cuda()
        input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
    return input_oh


def training_phase(step):
    epoch = 30
    for i in range(1, epoch + 1):
        print("Epoch {} / {}".format(i, epoch))

        b_CE_R, b_CE_N = 0, 0
        t_CE_R, t_CE_N = 0, 0

        for j, x in tqdm(enumerate(train_dl_dist), total=len(train_dl_dist)):
            d, r, n, c, c_r, c_n, r_density, n_density = x
            d, r, n, c = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float()

            r_density_lst, n_density_lst = r_density.float().cuda(), \
                                            n_density.float().cuda()

            d_oh = convert_to_one_hot(d, EVENT_DIMS)

            loss = train(d_oh, r_density_lst, n_density_lst)
            CE_R, CE_N = loss

            b_CE_R += CE_R
            b_CE_N += CE_N
        

        for j, x in tqdm(enumerate(val_dl_dist), total=len(val_dl_dist)):
            
            d, r, n, c, c_r, c_n, r_density, n_density = x
            d, r, n, c = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float()
            
            r_density_lst, n_density_lst = r_density.float().cuda(), \
                                            n_density.float().cuda()

            d_oh = convert_to_one_hot(d, EVENT_DIMS)

            loss = evaluate(d_oh, r_density_lst, n_density_lst)
            CE_R, CE_N = loss

            t_CE_R += CE_R
            t_CE_N += CE_N
        
        print("train loss by term: {:.5f}  {:.5f}".format(
            b_CE_R / len(train_dl_dist), 
            b_CE_N / len(train_dl_dist)
        ))
        print("test loss by term: {:.5f}  {:.5f}".format(
            t_CE_R / len(val_dl_dist), 
            t_CE_N / len(val_dl_dist)
        ))
    
    torch.save(model.cpu().state_dict(), "attribute_test.pt")


def evaluation_phase():

    model.load_state_dict(torch.load("attribute_test.pt"))
    if torch.cuda.is_available():
        model.cuda()
    
    def run(dl):
        
        t_CE_R, t_CE_N = 0, 0
        linear_r, linear_n = [], []

        for i, x in tqdm(enumerate(dl), total=len(dl)):
            r_density_lst, n_density_lst = [], []
            d, r, n, c, c_r, c_n, r_density, n_density = x
            d, r, n, c = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float()

            r_density_lst, n_density_lst = r_density.float().cuda(), \
                                            n_density.float().cuda()

            d_oh = convert_to_one_hot(d, EVENT_DIMS)

            r_predict, n_predict = model(d_oh)

            # calculate loss
            CE_R, CE_N = loss_function(r_predict, r_density_lst,
                                    n_predict, n_density_lst)
            
            from sklearn.linear_model import LinearRegression

            reg = LinearRegression().fit(r_predict.cpu().detach().numpy(), 
                                        r_density_lst.unsqueeze(-1).cpu().detach().numpy())
            linear_r.append(reg.score(r_predict.cpu().detach().numpy(), 
                    r_density_lst.unsqueeze(-1).cpu().detach().numpy()))

            reg = LinearRegression().fit(n_predict.cpu().detach().numpy(), 
                                        n_density_lst.unsqueeze(-1).cpu().detach().numpy())
            linear_n.append(reg.score(n_predict.cpu().detach().numpy(), 
                    n_density_lst.unsqueeze(-1).cpu().detach().numpy()))
            

            output = CE_R.item(), CE_N.item()

            # update
            t_CE_R += CE_R.item()
            t_CE_N += CE_N.item()
            
        print(sum(linear_r) / len(linear_r), sum(linear_n) / len(linear_n))
        
        # Print results
        print("CE: {:.4}  {:.4}".format(t_CE_R / len(dl), 
                                        t_CE_N / len(dl)))

    dl = DataLoader(test_ds_dist, batch_size=128, shuffle=False, num_workers=0)
    run(dl)


training_phase(step)
evaluation_phase()

