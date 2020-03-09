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
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, z, labels):
        # Concatenate label embedding and image to produce input
        z_input = torch.cat([z, labels], dim=-1)
        z_gen = self.dropout(self.bn1(self.lrelu(self.linear(z_input))))
        z_gen = self.dropout(self.bn2(self.linear2(z_gen)))

        gate = nn.Sigmoid()(z_gen)
        z_out = gate * z_gen + (1-gate) * z
        return z_out


class Discriminator(nn.Module):
    def __init__(self, latent_dim, preset_dim=2):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(latent_dim + preset_dim, 32)
        self.linear2 = nn.Linear(32, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.bn = nn.BatchNorm1d(32)

    def forward(self, z, labels):
        # Concatenate label embedding and image to produce input
        z_input = torch.cat([z, labels], dim=-1)
        validity = self.dropout(self.lrelu(self.bn(self.linear(z_input))))
        validity = nn.Sigmoid()(self.linear2(validity))
        
        return validity


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

    model.load_state_dict(torch.load("params/music_attr_vae_reg_vgm.pt"))
    # freeze model
    for p in model.parameters():
        p.requires_grad = False


    # ===================== TRAINING MODEL ==================== #
    # Loss functions
    adversarial_loss = torch.nn.BCELoss()

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
        adversarial_loss.cuda()

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

    def convert_to_one_hot(input, dims):
        if len(input.shape) > 1:
            input_oh = torch.zeros((input.shape[0], input.shape[1], dims)).cuda()
            input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
        else:
            input_oh = torch.zeros((input.shape[0], dims)).cuda()
            input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
        return input_oh


    def pass_model(x):
        d, r, n, c, a, v, r_density, n_density = x
        d, r, n, c = d.cuda().long(), r.cuda().long(), \
                        n.cuda().long(), c.cuda().float()
        
        d_oh = convert_to_one_hot(d, EVENT_DIMS)
        # get q(z|x)
        dis_r, dis_n = model.encoder(d_oh, None, None, c)
        z_r, z_n = dis_r.sample(), dis_n.sample()

        a_label = torch.Tensor(np.eye(2, dtype='uint8')[a.int().cpu().detach().numpy()]).cuda()

        return z_r, z_n, a_label


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
        

    d_loss_lst, g_loss_lst = [], []
    for epoch in range(args["n_epochs"]):

        for i, x in enumerate(vgm_train_dl_dist):
            
            z_r_q, z_n_q, a_label = pass_model(x)

            # Adversarial ground truths
            valid = Variable(FloatTensor(z_r_q.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(z_r_q.shape[0], 1).fill_(0.0), requires_grad=False)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_Gr.zero_grad()
            optimizer_Gn.zero_grad()

            # Sample noise and labels as generator input
            z_r_p, z_n_p = sample_prior(z_r_q)

            # Generate a batch of images
            z_r_p_gen, z_n_p_gen = generator_r(z_r_p, a_label), generator_n(z_n_p, a_label)

            # Loss measures generator's ability to fool the discriminator
            validity_r, validity_n = discriminator_r(z_r_p_gen, a_label), discriminator_n(z_n_p_gen, a_label)
            validity_int_r, validity_int_n = validity_int(validity_r), validity_int(validity_n)
            g_loss = adversarial_loss(validity_r, valid) + adversarial_loss(validity_n, valid)
            g_acc = (accuracy_score(np.ones(valid.shape[0]), validity_int_r) + \
                    accuracy_score(np.ones(valid.shape[0]), validity_int_n)) / 2

            g_loss.backward()
            optimizer_Gr.step()
            optimizer_Gn.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_Dr.zero_grad()
            optimizer_Dn.zero_grad()

            # Loss for q(z|x)
            validity_q_r, validity_q_n = discriminator_r(z_r_q, a_label), discriminator_n(z_n_q, a_label)
            validity_int_qr, validity_int_qn = validity_int(validity_q_r), validity_int(validity_q_n)
            d_real_acc = (accuracy_score(np.ones(valid.shape[0]), validity_int_qr) + \
                            accuracy_score(np.ones(valid.shape[0]), validity_int_qn)) / 2
            d_real_loss = adversarial_loss(validity_q_r, valid) + adversarial_loss(validity_q_n, valid)

            # Loss for p(z)
            validity_p_r, validity_p_n = discriminator_r(z_r_p, a_label), discriminator_n(z_n_p, a_label)
            validity_int_pr, validity_int_pn = validity_int(validity_p_r), validity_int(validity_p_n)
            d_fake_acc = (accuracy_score(np.zeros(valid.shape[0]), validity_int_pr) + \
                            accuracy_score(np.zeros(valid.shape[0]), validity_int_pn)) / 2
            d_fake_loss = adversarial_loss(validity_p_r, fake) + adversarial_loss(validity_p_n, fake)

            # Loss for G(p(z), y)
            validity_g_r, validity_g_n = discriminator_r(z_r_p_gen, a_label), discriminator_n(z_n_p_gen, a_label)
            validity_int_gr, validity_int_gn = validity_int(validity_g_r), validity_int(validity_g_n)
            d_gen_acc = (accuracy_score(np.zeros(valid.shape[0]), validity_int_gr) + \
                            accuracy_score(np.zeros(valid.shape[0]), validity_int_gn)) / 2
            d_gen_loss = adversarial_loss(validity_g_r.detach(), fake) + adversarial_loss(validity_g_n.detach(), fake)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss + d_gen_loss) / 3
            d_acc = (d_real_acc + d_fake_acc + d_gen_acc) / 3

            d_loss.backward(retain_graph=True)
            optimizer_Dr.step()
            optimizer_Dn.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D acc: %f] [G acc: %f]"
                % (epoch, args["n_epochs"], i, len(vgm_train_dl_dist), d_loss.item(), g_loss.item(), d_acc, g_acc)
            )

            d_loss_lst.append(d_loss.item())
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