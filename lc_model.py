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

class Generator(nn.Module):
    def __init__(self, latent_dim, preset_dim=2):
        super(Generator, self).__init__()
        self.linear = nn.Linear(latent_dim + preset_dim, latent_dim)
        self.linear2 = nn.Linear(latent_dim, latent_dim)
        self.linear3 = nn.Linear(latent_dim, latent_dim)
        self.gate_linear = nn.Linear(latent_dim, latent_dim)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        self.bn4 = nn.BatchNorm1d(latent_dim)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.label_emb = nn.Embedding(preset_dim, preset_dim)

    def forward(self, z, labels):
        # Concatenate label embedding and image to produce input
        label_embed = self.label_emb(labels.long()).squeeze()
        if len(label_embed.shape) < 2: label_embed = label_embed.unsqueeze(0)
        z_input = torch.cat([z, label_embed], dim=-1)
        z_hid = self.dropout(self.bn1(self.lrelu1(self.linear(z_input))))
        z_hid = self.dropout(self.bn2(self.lrelu2(self.linear2(z_hid))))
        z_gen = self.dropout(self.bn3(self.linear3(z_hid)))
        gate = self.dropout(self.bn4(self.gate_linear(z_hid)))

        gate = nn.Sigmoid()(gate)
        z_out = gate * z_gen + (1-gate) * z
        return z_out


class Discriminator(nn.Module):
    def __init__(self, latent_dim, preset_dim=0):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(latent_dim + preset_dim, 32)
        self.linear2 = nn.Linear(32, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.bn = nn.BatchNorm1d(32)

    def forward(self, z):
        # Concatenate label embedding and image to produce input
        validity = self.dropout(self.lrelu(self.bn(self.linear(z))))
        validity = nn.Sigmoid()(self.linear2(validity))
        
        return validity


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
    print("Loading Yamaha...")
    is_shuffle = True
    batch_size = args["batch_size"]
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
    vgm_test_dl_dist = DataLoader(vgm_test_ds_dist, batch_size=32, shuffle=is_shuffle, num_workers=0)
    print(len(vgm_train_ds_dist), len(vgm_val_ds_dist), len(vgm_test_ds_dist))
    print()


    # ===================== PRETRAINED MODEL ==================== #
    # model dimensions
    EVENT_DIMS = 342
    RHYTHM_DIMS = 3
    NOTE_DIMS = 16
    TEMPO_DIMS = 264
    VELOCITY_DIMS = 126
    CHROMA_DIMS = 24

    model = MusicAttrRegVAE(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
                            tempo_dims=TEMPO_DIMS, velocity_dims=VELOCITY_DIMS, chroma_dims=CHROMA_DIMS,
                            hidden_dims=args['hidden_dim'], z_dims=args['z_dim'], 
                            n_step=args['time_step'])

    model.load_state_dict(torch.load("params/music_attr_vae_reg_110220.pt"))
    # freeze model
    for p in model.parameters():
        p.requires_grad = False


    # ===================== TRAINING MODEL ==================== #
    # Loss functions
    def w_binary_cross_entropy(output, target, weights=None):
        if weights is not None:
            assert len(weights) == 2
            loss = weights[1] * (target * torch.log(output)) + \
                weights[0] * ((1 - target) * torch.log(1 - output))
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
        return torch.neg(torch.mean(loss))
    
    w = torch.Tensor([1/0.3, 1/0.7])
    adversarial_loss = w_binary_cross_entropy

    # Initialize generator and discriminator
    generator_r = Generator(latent_dim=args["z_dim"])
    generator_n = Generator(latent_dim=args["z_dim"])
    discriminator_r = Discriminator(latent_dim=args["z_dim"])
    discriminator_n = Discriminator(latent_dim=args["z_dim"])

    if cuda:
        model.cuda()
        generator_r.cuda()
        generator_n.cuda()
        discriminator_r.cuda()
        discriminator_n.cuda()
        # adversarial_loss.cuda()

    # Optimizers
    optimizer_Gr = torch.optim.Adam(generator_r.parameters(), lr=args["lr"])
    optimizer_Gn = torch.optim.Adam(generator_n.parameters(), lr=args["lr"])
    optimizer_Dr = torch.optim.Adam(discriminator_r.parameters(), lr=args["lr"])
    optimizer_Dn = torch.optim.Adam(discriminator_n.parameters(), lr=args["lr"])

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # ----------
    #  Training
    # ----------

    def pass_model(x, is_a=True):
        d, r, n, c, a, v, r_density, n_density = x
        d, r, n, c = d.cuda().long(), r.cuda().long(), \
                        n.cuda().long(), c.cuda().float()
        
        d_oh = convert_to_one_hot(d, EVENT_DIMS)
        # get q(z|x)
        dis_r, dis_n = model.encoder(d_oh, None, None, c)
        z_r, z_n = dis_r.sample(), dis_n.sample()

        if is_a:
            # a_label = torch.Tensor(np.eye(2, dtype='uint8')[a.int().cpu().detach().numpy()]).cuda()
            a_label = torch.Tensor(a).cuda().unsqueeze(-1)
        else:
            a_label = 0

        return z_r, z_n, a_label, dis_r, dis_n


    def sample_prior(z):
        N = torch.distributions.Normal(torch.zeros(z.size()), torch.ones(z.size()))
        if torch.cuda.is_available():
            N.loc = N.loc.cuda()
            N.scale = N.scale.cuda()
        
        z_r = N.sample()
        z_n = N.sample()

        return z_r, z_n
    

    def validity_int(validity):
        res = np.zeros(validity.shape[0],)
        res[np.where(validity.cpu().detach().numpy().squeeze() > 0.5)] = 1
        return res
    

    def distance_reg_loss(dis_var, z_gen, z):
        mean_var = torch.mean(dis_var, dim=0)
        return torch.mean(torch.log(1 + F.mse_loss(z_gen, z, size_average=False, reduce=False)) / mean_var)
        

    d_loss_lst, g_loss_lst = [], []
    for epoch in range(args["n_epochs"]):

        # supervised train first
        for i, x in enumerate(vgm_train_dl_dist):
            
            z_r_q, z_n_q, a_label, _, _ = pass_model(x)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_Dr.zero_grad()
            optimizer_Dn.zero_grad()

            # Loss for q(z|x)
            validity_q_r, validity_q_n = discriminator_r(z_r_q), discriminator_n(z_n_q)
            validity_int_qr, validity_int_qn = validity_int(validity_q_r), validity_int(validity_q_n)
            d_real_acc = (accuracy_score(a_label.cpu().detach().numpy().squeeze(), validity_int_qr) + \
                            accuracy_score(a_label.cpu().detach().numpy().squeeze(), validity_int_qn)) / 2
            d_real_loss = adversarial_loss(validity_q_r, a_label, weights=w) + adversarial_loss(validity_q_n, a_label, weights=w)
            
            # Total discriminator loss
            d_loss = d_real_loss
            d_acc = d_real_acc

            d_loss.backward()
            optimizer_Dr.step()
            optimizer_Dn.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [D acc: %f]"
                % (epoch, args["n_epochs"], i, len(vgm_train_dl_dist), d_loss.item(), d_acc)
            )

            d_loss_lst.append(d_loss.item())
        
        print()
        # supervised train first
        for i, x in enumerate(vgm_test_dl_dist):
            
            z_r_q, z_n_q, a_label, _, _ = pass_model(x)

            # ---------------------
            #  Evaluate Discriminator
            # ---------------------

            # Loss for q(z|x)
            validity_q_r, validity_q_n = discriminator_r(z_r_q), discriminator_n(z_n_q)
            validity_int_qr, validity_int_qn = validity_int(validity_q_r), validity_int(validity_q_n)
            d_real_acc = (accuracy_score(a_label.cpu().detach().numpy().squeeze(), validity_int_qr) + \
                            accuracy_score(a_label.cpu().detach().numpy().squeeze(), validity_int_qn)) / 2
            d_real_loss = adversarial_loss(validity_q_r, a_label, weights=w) + adversarial_loss(validity_q_n, a_label, weights=w)
            
            # Total discriminator loss
            d_loss = d_real_loss
            d_acc = d_real_acc

            print(
                "[Epoch %d/%d] [Batch %d/%d] [DT loss: %f] [DT acc: %f]"
                % (epoch, args["n_epochs"], i, len(vgm_train_dl_dist), d_loss.item(), d_acc)
            )

            d_loss_lst.append(d_loss.item())

        print()
        
        # unsupervised train
        for i, x in enumerate(train_dl_dist):

            z_r_q, z_n_q, _, dis_r, dis_n = pass_model(x, is_a=False)

            # Adversarial ground truths
            valid = Variable(FloatTensor(z_r_q.shape[0], 1).fill_(0.0), requires_grad=False)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_Gr.zero_grad()
            optimizer_Gn.zero_grad()

            # Generate a batch of images
            z_r_q_gen, z_n_q_gen = generator_r(z_r_q, valid), generator_n(z_n_q, valid)

            # Critic loss measures generator's ability to generate exact
            validity_r, validity_n = discriminator_r(z_r_q_gen), discriminator_n(z_n_q_gen)
            validity_int_r, validity_int_n = validity_int(validity_r), validity_int(validity_n)

            g_loss = adversarial_loss(validity_r, valid, weights=w) + adversarial_loss(validity_n, valid, weights=w)
            g_loss += 0.5 * (distance_reg_loss(dis_r.variance, z_r_q_gen, z_r_q) + \
                      distance_reg_loss(dis_n.variance, z_n_q_gen, z_n_q))
            g_acc = (accuracy_score(valid.cpu().detach().numpy().squeeze(), validity_int_r) + \
                    accuracy_score(valid.cpu().detach().numpy().squeeze(), validity_int_n)) / 2
            
            g_loss.backward()
            optimizer_Gr.step()
            optimizer_Gn.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [G acc: %f]"
                % (epoch, args["n_epochs"], i, len(train_dl_dist), g_loss.item(), g_acc)
            )
            g_loss_lst.append(g_loss.item())
        
        print()

        # unsupervised train
        for i, x in enumerate(train_dl_dist):

            z_r_q, z_n_q, _, dis_r, dis_n = pass_model(x, is_a=False)

            # Adversarial ground truths
            fake = Variable(FloatTensor(z_r_q.shape[0], 1).fill_(0.0), requires_grad=False)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_Gr.zero_grad()
            optimizer_Gn.zero_grad()

            # Generate a batch of images
            z_r_q_gen, z_n_q_gen = generator_r(z_r_q, fake), generator_n(z_n_q, fake)

            # Critic loss measures generator's ability to generate exact
            validity_r, validity_n = discriminator_r(z_r_q_gen), discriminator_n(z_n_q_gen)
            validity_int_r, validity_int_n = validity_int(validity_r), validity_int(validity_n)
            
            g_loss = adversarial_loss(validity_r, fake, weights=w) + adversarial_loss(validity_n, fake, weights=w)
            g_loss += 0.5 * (distance_reg_loss(dis_r.variance, z_r_q_gen, z_r_q) + \
                      distance_reg_loss(dis_n.variance, z_n_q_gen, z_n_q))
            g_acc = (accuracy_score(fake.cpu().detach().numpy().squeeze(), validity_int_r) + \
                    accuracy_score(fake.cpu().detach().numpy().squeeze(), validity_int_n)) / 2

            g_loss.backward()
            optimizer_Gr.step()
            optimizer_Gn.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [GT loss: %f] [GT acc: %f]"
                % (epoch, args["n_epochs"], i, len(train_dl_dist), g_loss.item(), g_acc)
            )
            g_loss_lst.append(g_loss.item())
            batches_done = epoch * len(vgm_train_dl_dist) + i



    plt.plot(d_loss_lst)
    plt.plot(g_loss_lst)
    plt.savefig("lc_loss.png")
    plt.close()

    torch.save(generator_r.cpu().state_dict(), 'lc_params/generator_r.pt')
    torch.save(generator_n.cpu().state_dict(), 'lc_params/generator_n.pt')
    torch.save(discriminator_r.cpu().state_dict(), 'lc_params/discriminator_r.pt')
    torch.save(discriminator_n.cpu().state_dict(), 'lc_params/discriminator_n.pt')