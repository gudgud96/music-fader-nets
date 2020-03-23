import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from sklearn.metrics import accuracy_score

from tqdm import tqdm
from ptb_v2 import *
from uni_model_2 import *

with open('lc_model_config.json') as f:
    args = json.load(f)
if not os.path.isdir('lc_params'):
    os.mkdir('lc_params')

cuda = True if torch.cuda.is_available() else False

# =================== LC MODEL PARTS ==================== #

class Discriminator(nn.Module):
    def __init__(self, roll_dims=342):
        super(Discriminator, self).__init__()
        self.gru = nn.GRU(roll_dims, 128, batch_first=True)
        self.linear = nn.Linear(128, 32)
        self.linear2 = nn.Linear(32, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.bn = nn.BatchNorm1d(32)

    def forward(self, x):
        h = self.gru(x)[-1]
        h = h.transpose_(0, 1).contiguous().view(h.size(0), -1)
        h = self.dropout(self.lrelu(self.bn(self.linear(h))))
        res = nn.Sigmoid()(self.linear2(h))
        
        return res


def convert_to_one_hot(input, dims):
    if len(input.shape) > 1:
        input_oh = torch.zeros((input.shape[0], input.shape[1], dims)).cuda()
        input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
    else:
        input_oh = torch.zeros((input.shape[0], dims)).cuda()
        input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
    return input_oh


if __name__ == "__main__":
    # ======================== DATALOADERS ======================== #
    is_shuffle = True

    # vgmidi dataloaders
    print("Loading VGMIDI...")
    data_lst, rhythm_lst, note_density_lst, arousal_lst, valence_lst, chroma_lst = get_vgmidi()
    vgm_train_ds_dist = MusicAttrDataset3(data_lst, rhythm_lst, note_density_lst, 
                                    chroma_lst, arousal_lst, valence_lst, mode="train")
    vgm_train_dl_dist = DataLoader(vgm_train_ds_dist, batch_size=32, shuffle=is_shuffle, num_workers=0)
    vgm_val_ds_dist = MusicAttrDataset3(data_lst, rhythm_lst, note_density_lst, 
                                    chroma_lst, arousal_lst, valence_lst, mode="val")
    vgm_val_dl_dist = DataLoader(vgm_val_ds_dist, batch_size=32, shuffle=is_shuffle, num_workers=0)
    vgm_test_ds_dist = MusicAttrDataset3(data_lst, rhythm_lst, note_density_lst, 
                                    chroma_lst, arousal_lst, valence_lst, mode="test")
    vgm_test_dl_dist = DataLoader(vgm_test_ds_dist, batch_size=len(data_lst), shuffle=is_shuffle, num_workers=0)
    print(len(vgm_train_ds_dist), len(vgm_val_ds_dist), len(vgm_test_ds_dist))
    print()

    
    # ===================== TRAINING MODEL ==================== #
    # Loss functions
    criterion = nn.BCELoss()
    arousal_clf = Discriminator()

    if cuda:
        arousal_clf.cuda()

    # Optimizers
    optimizer = torch.optim.Adam(arousal_clf.parameters(), lr=args["lr"])
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # ----------
    #  Training
    # ----------

    loss_lst = []
    for epoch in range(args["n_epochs"]):

        # supervised train first
        for i, x in enumerate(vgm_train_dl_dist):

            optimizer.zero_grad()
            d, r, n, c, a, v, r_density, n_density = x
            d, r, n, c = d.cuda().long(), r.cuda().long(), \
                            n.cuda().long(), c.cuda().float()
            
            d_oh = convert_to_one_hot(d, 342)
            out_logits = arousal_clf(d_oh)
            loss = criterion(out_logits.squeeze(), a.cuda())

            loss.backward()
            optimizer.step()

            out_predict = out_logits.clone().squeeze().cpu().detach().numpy()
            out_predict[out_predict >= 0.5] = 1
            out_predict[out_predict < 0.5] = 0
            acc = accuracy_score(a.cpu().detach().numpy(), out_predict)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [D acc: %f]"
                % (epoch, args["n_epochs"], i, len(vgm_train_dl_dist), loss.item(), acc)
            )

            loss_lst.append(loss.item())
        
        print()
        # supervised train first
        for i, x in enumerate(vgm_test_dl_dist):
            
            d, r, n, c, a, v, r_density, n_density = x
            d, r, n, c = d.cuda().long(), r.cuda().long(), \
                            n.cuda().long(), c.cuda().float()
            
            d_oh = convert_to_one_hot(d, 342)
            out_logits = arousal_clf(d_oh)
            loss = criterion(out_logits.squeeze(), a.cuda())

            out_predict = out_logits.clone().squeeze().cpu().detach().numpy()
            out_predict[out_predict >= 0.5] = 1
            out_predict[out_predict < 0.5] = 0
            acc = accuracy_score(a.cpu().detach().numpy(), out_predict)

            print(
                "[Test Epoch %d/%d] [Batch %d/%d] [D loss: %f] [D acc: %f]"
                % (epoch, args["n_epochs"], i, len(vgm_test_dl_dist), loss.item(), acc)
            )
            

    plt.plot(loss_lst)
    plt.savefig("clf_loss.png")
    plt.close()

    torch.save(arousal_clf.cpu().state_dict(), 'lc_params/arousal_clf.pt')