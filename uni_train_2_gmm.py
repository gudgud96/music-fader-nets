import json
import torch
import os
import numpy as np
from gmm_model import *
# from data_loader import MusicArrayLoader
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split
from adversarial_test import *
from polyphonic_event_based_v2 import parse_pretty_midi

from ptb_v2 import *


# some initialization
with open('gmm_model_config.json') as f:
    args = json.load(f)
if not os.path.isdir('log'):
    os.mkdir('log')
if not os.path.isdir('params'):
    os.mkdir('params')
save_path = 'params/{}.pt'.format(args['name'])

from datetime import datetime
timestamp = str(datetime.now())
save_path_timing = 'params/{}.pt'.format(args['name'] + "_" + timestamp)

# ====================== MODELS ===================== #
# model dimensions
EVENT_DIMS = 342
RHYTHM_DIMS = 3
NOTE_DIMS = 16
TEMPO_DIMS = 264
VELOCITY_DIMS = 126
CHROMA_DIMS = 24

is_adversarial = args["is_adversarial"]
print("Is adversarial: {}".format(is_adversarial))

model = MusicAttrRegGMVAE(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
                        tempo_dims=TEMPO_DIMS, velocity_dims=VELOCITY_DIMS, chroma_dims=CHROMA_DIMS,
                        hidden_dims=args['hidden_dim'], z_dims=args['z_dim'], 
                        n_step=args['time_step'],
                        n_component=2)      # 2 clusters at the moment

if os.path.exists(save_path):
    print("Loading {}".format(save_path))
    model.load_state_dict(torch.load(save_path))
else:
    print("Save path: {}".format(save_path))

optimizer = optim.Adam(model.parameters(), lr=args['lr'])

if torch.cuda.is_available():
    print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    model.cuda()
else:
    print('CPU mode')

step, pre_epoch = 0, 0
batch_size = args["batch_size"]
model.train()

# ====================== DATALOADERS ===================== #
# dataloaders
print("Loading Yamaha...")
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

is_class = args["is_class"]
is_res = args["is_res"]
# end of initialization


# ====================== TRAINING ===================== #

def std_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N


def loss_function(out, d,
                r_out, r,
                n_out, n,
                dis,
                qy_x_out,
                logLogit_out,
                step,
                beta=.1,
                is_supervised=False,
                y_label=None):
    '''
    Following loss function defined for GMM-VAE:
    Unsupervised: E[log p(x|z)] - sum{l} q(y_l|X) * KL[q(z|x) || p(z|y_l)] - KL[q(y|x) || p(y)]
    Supervised: E[log p(x|z)] - KL[q(z|x) || p(z|y)]
    '''

    # Reconstruction loss
    CE_X = F.nll_loss(out.view(-1, out.size(-1)),
                    d.view(-1), reduction='mean')
    CE_R = F.nll_loss(r_out.view(-1, r_out.size(-1)),
                    r.view(-1), reduction='mean')
    CE_N = F.nll_loss(n_out.view(-1, n_out.size(-1)),
                    n.view(-1), reduction='mean')

    CE = 5 * CE_X + CE_R + CE_N

    # package output
    dis_r, dis_n = dis
    qy_x_r, qy_x_n = qy_x_out
    logLogit_qy_x_r, logLogit_qy_x_n = logLogit_out
    
    # KLD latent and class loss
    kld_lat_r_total, kld_lat_n_total = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
    kld_cls_r, kld_cls_n = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()

    if not is_supervised:
        # KL latent loss
        n_component = qy_x_r.shape[-1]
        
        for k in torch.arange(0, n_component):       # number of components
            # infer current p(z|y)
            mu_pz_y_r, var_pz_y_r = model.mu_r_lookup(k.cuda()), model.logvar_r_lookup(k.cuda()).exp_()
            dis_pz_y_r = Normal(mu_pz_y_r, var_pz_y_r)
            kld_lat_r = torch.sum(kl_divergence(dis_r, dis_pz_y_r), dim=-1)
            kld_lat_r *= qy_x_r[:, k]

            mu_pz_y_n, var_pz_y_n = model.mu_n_lookup(k.cuda()), model.logvar_n_lookup(k.cuda()).exp_()
            dis_pz_y_n = Normal(mu_pz_y_n, var_pz_y_n)
            kld_lat_n = torch.sum(kl_divergence(dis_n, dis_pz_y_n), dim=-1)
            kld_lat_n *= qy_x_n[:, k]

            kld_lat_r_total += kld_lat_r.mean()
            kld_lat_n_total += kld_lat_n.mean()
        
        # KL class loss --> KL[q(y|x) || p(y)] = H(q(y|x)) - log p(y)
        def entropy(qy_x, logLogit_qy_x):
            return torch.sum(qy_x * torch.nn.functional.log_softmax(logLogit_qy_x, dim=1), dim=1)
        
        h_qy_x_r = entropy(qy_x_r, logLogit_qy_x_r)
        h_qy_x_n = entropy(qy_x_n, logLogit_qy_x_n)
        kld_cls_r = (h_qy_x_r - np.log(1 / n_component)).mean()
        kld_cls_n = (h_qy_x_n - np.log(1 / n_component)).mean()
    
    else:
        mu_pz_y_r, var_pz_y_r = model.mu_r_lookup(y_label.cuda().long()), model.logvar_r_lookup(y_label.cuda().long()).exp_()
        dis_pz_y_r = Normal(mu_pz_y_r, var_pz_y_r)
        kld_lat_r = torch.sum(kl_divergence(dis_r, dis_pz_y_r), dim=-1)

        mu_pz_y_n, var_pz_y_n = model.mu_n_lookup(y_label.cuda().long()), model.logvar_n_lookup(y_label.cuda().long()).exp_()
        dis_pz_y_n = Normal(mu_pz_y_n, var_pz_y_n)
        kld_lat_n = torch.sum(kl_divergence(dis_n, dis_pz_y_n), dim=-1)

        kld_lat_r_total, kld_lat_n_total = kld_lat_r.mean(), kld_lat_n.mean()
        
    # anneal beta
    # beta0 = min(step / 3000 * beta, beta)  
    beta0 = beta
    loss = CE + kld_lat_r_total + kld_lat_n_total + kld_cls_r + kld_cls_n
    return loss, CE_X, CE_R, CE_N, kld_lat_r_total, kld_lat_n_total, kld_cls_r, kld_cls_n


def latent_regularized_loss_function(z_out, r, n, a=None, c=None):
    return latent_regularized_loss_function_v1(z_out, r, n, a=a)
    # return latent_regularized_loss_function_glsr(z_out, r, n, c)


def latent_regularized_loss_function_v1(z_out, r, n, a=None):
    # regularization loss - Pati et al. 2019
    z_r, z_n = z_out

    z_r_new = z_r
    z_n_new = z_n

    # rhythm regularized
    r_density = r
    D_attr_r = torch.from_numpy(np.subtract.outer(r_density, r_density)).cuda().float()
    D_z_r = z_r_new[:, 0].reshape(-1, 1) - z_r_new[:, 0]
    l_r = torch.nn.MSELoss(reduction="mean")(torch.tanh(D_z_r), torch.sign(D_attr_r))
        
    n_density = n
    D_attr_n = torch.from_numpy(np.subtract.outer(n_density, n_density)).cuda().float()
    D_z_n = z_n_new[:, 0].reshape(-1, 1) - z_n_new[:, 0]
    l_n = torch.nn.MSELoss(reduction="mean")(torch.tanh(D_z_n), torch.sign(D_attr_n))

    if not(a is None):
        arousal = a
        D_attr_a = torch.from_numpy(np.subtract.outer(arousal, arousal)).cuda().float()
        D_z_h = z_h[:, 0].reshape(-1, 1) - z_h[:, 0]
        l_a = torch.nn.MSELoss(reduction="mean")(torch.tanh(D_z_h), torch.sign(D_attr_a))
    else:
        l_a = 0

    return l_r, l_n, l_a


def train(step, d_oh, r_oh, n_oh,
          d, r, n, c, c_r, c_n, c_r_oh, c_n_oh, r_density, n_density,
          is_supervised=False, y_label=None):
    
    optimizer.zero_grad()
    res = model(d_oh, r_oh, n_oh, c, c_r_oh, c_n_oh, is_class=is_class, is_res=is_res)

    # package output
    output, dis, z_out, logLogit_out, qy_x_out, y_out = res
    out, r_out, n_out, _, _ = output
    z_r, z_n = z_out

    # calculate gmm loss
    loss, CE_X, CE_R, CE_N, kld_lat_r_total, kld_lat_n_total, \
        kld_cls_r, kld_cls_n = loss_function(out, d,
                                        r_out, r,
                                        n_out, n,
                                        dis,
                                        qy_x_out,
                                        logLogit_out,
                                        step,
                                        beta=args['beta'],
                                        is_supervised=is_supervised,
                                        y_label=y_label)
    
    # calculate latent regularization loss
    l_r, l_n = torch.Tensor([0]), torch.Tensor([0])
    l_r, l_n, _ = latent_regularized_loss_function(z_out, r_density, n_density, c=c)
    loss += l_r + l_n
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    step += 1

    kld_latent = kld_lat_r_total + kld_lat_n_total
    kld_class = kld_cls_r + kld_cls_n
    
    output = loss.item(), CE_X.item(), CE_R.item(), CE_N.item(), l_r.item(), l_n.item(), kld_latent.item(), kld_class.item()
    return step, output


def evaluate(step, d_oh, r_oh, n_oh,
             d, r, n, c, c_r, c_n, c_r_oh, c_n_oh, r_density, n_density,
             is_supervised=False, y_label=None):
    
    res = model(d_oh, r_oh, n_oh, c, c_r_oh, c_n_oh, is_class=is_class, is_res=is_res)

    # package output
    output, dis, z_out, logLogit_out, qy_x_out, y_out = res
    out, r_out, n_out, _, _ = output
    z_r, z_n = z_out

    # calculate gmm loss
    loss, CE_X, CE_R, CE_N, kld_lat_r_total, kld_lat_n_total, \
        kld_cls_r, kld_cls_n = loss_function(out, d,
                                        r_out, r,
                                        n_out, n,
                                        dis,
                                        qy_x_out,
                                        logLogit_out,
                                        step,
                                        beta=args['beta'],
                                        is_supervised=is_supervised,
                                        y_label=y_label)
    
    # calculate latent regularization loss
    l_r, l_n = torch.Tensor([0]), torch.Tensor([0])
    l_r, l_n, _ = latent_regularized_loss_function(z_out, r_density, n_density, c=c)
    loss += l_r + l_n

    kld_latent = kld_lat_r_total + kld_lat_n_total
    kld_class = kld_cls_r + kld_cls_n
    
    output = loss.item(), CE_X.item(), CE_R.item(), CE_N.item(), l_r.item(), l_n.item(), kld_latent.item(), kld_class.item()
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

    for i in range(1, args['n_epochs'] + 1):
        print("Epoch {} / {}".format(i, args['n_epochs']))

        batch_loss, batch_test_loss = 0, 0
        b_CE_X, b_CE_R, b_CE_N, b_CE_C, b_CE_T, b_CE_V = 0, 0, 0, 0, 0, 0
        t_CE_X, t_CE_R, t_CE_N, t_CE_C, t_CE_T, t_CE_V = 0, 0, 0, 0, 0, 0
        b_l_r, b_l_n, t_l_r, t_l_n = 0, 0, 0, 0
        b_l_adv_r, b_l_adv_n, t_l_adv_r, t_l_adv_n = 0, 0, 0, 0

        # train on yamaha
        # for j, x in tqdm(enumerate(train_dl_dist), total=len(train_dl_dist)):

        #     d, r, n, c, c_r, c_n, r_density, n_density = x
        #     d, r, n, c = d.cuda().long(), r.cuda().long(), \
        #                  n.cuda().long(), c.cuda().float()
        #     c_r, c_n, = c_r.cuda().long(), c_n.cuda().long()

        #     d_oh = convert_to_one_hot(d, EVENT_DIMS)
        #     r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
        #     n_oh = convert_to_one_hot(n, NOTE_DIMS)

        #     c_r_oh = convert_to_one_hot(c_r, 3)
        #     c_n_oh = convert_to_one_hot(c_n, 3)

        #     step, loss = train(step, d_oh, r_oh, n_oh, d, r, n, c, 
        #                        c_r, c_n, c_r_oh, c_n_oh, r_density, n_density)
        #     loss, CE_X, CE_R, CE_N, l_r, l_n, kld_latent, kld_class = loss
        #     batch_loss += loss

        #     inputs = b_CE_X, b_CE_R, b_CE_N
        #     updates = CE_X, CE_R, CE_N
        #     outputs = []
        #     for idx in range(len(inputs)):
        #         outputs.append(inputs[idx] + updates[idx])
        #     b_CE_X, b_CE_R, b_CE_N = outputs
        #     b_l_r += l_r
        #     b_l_n += l_n
        #     b_l_adv_r += kld_latent
        #     b_l_adv_n += kld_class 

        # # evaluate on yamaha
        # for j, x in tqdm(enumerate(val_dl_dist), total=len(val_dl_dist)):
            
        #     d, r, n, c, c_r, c_n, r_density, n_density = x
        #     d, r, n, c = d.cuda().long(), r.cuda().long(), \
        #                  n.cuda().long(), c.cuda().float()
        #     c_r, c_n, = c_r.cuda().long(), c_n.cuda().long()

        #     d_oh = convert_to_one_hot(d, EVENT_DIMS)
        #     r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
        #     n_oh = convert_to_one_hot(n, NOTE_DIMS)

        #     c_r_oh = convert_to_one_hot(c_r, 3)
        #     c_n_oh = convert_to_one_hot(c_n, 3)

        #     loss = evaluate(step - 1, d_oh, r_oh, n_oh, d, r, n, c, 
        #                     c_r, c_n, c_r_oh, c_n_oh, r_density, n_density)

        #     loss, CE_X, CE_R, CE_N, l_r, l_n, kld_latent, kld_class = loss
        #     batch_test_loss += loss

        #     inputs = t_CE_X, t_CE_R, t_CE_N
        #     updates = CE_X, CE_R, CE_N
        #     outputs = []
        #     for idx in range(len(inputs)):
        #         outputs.append(inputs[idx] + updates[idx])
        #     t_CE_X, t_CE_R, t_CE_N = outputs
        #     t_l_r += l_r
        #     t_l_n += l_n
        #     t_l_adv_r += kld_latent
        #     t_l_adv_n += kld_class
        
        # print('batch loss: {:.5f}  {:.5f}'.format(batch_loss / len(train_dl_dist),
        #                                           batch_test_loss / len(val_dl_dist)))
        # print("Data, Rhythm, Note, Chroma")
        # print("train loss by term: {:.5f}  {:.5f}  {:.5f}  {:.5f}  {:.5f}  {:.5f}  {:.5f}".format(
        #     b_CE_X / len(train_dl_dist), b_CE_R / len(train_dl_dist), 
        #     b_CE_N / len(train_dl_dist),
        #     b_l_r / len(train_dl_dist), b_l_n / len(train_dl_dist),
        #     b_l_adv_r / len(train_dl_dist), b_l_adv_n / len(train_dl_dist)
        # ))
        # print("test loss by term: {:.5f}  {:.5f}  {:.5f}  {:.5f}  {:.5f}  {:.5f}  {:.5f}".format(
        #     t_CE_X / len(val_dl_dist), t_CE_R / len(val_dl_dist), 
        #     t_CE_N / len(val_dl_dist),
        #     t_l_r / len(val_dl_dist), t_l_n / len(val_dl_dist),
        #     t_l_adv_r / len(val_dl_dist), t_l_adv_n / len(val_dl_dist)
        # ))

        # =================== TRAIN VGMIDI ======================== #

        batch_loss, batch_test_loss = 0, 0
        b_CE_X, b_CE_R, b_CE_N, b_CE_C, b_CE_T, b_CE_V = 0, 0, 0, 0, 0, 0
        t_CE_X, t_CE_R, t_CE_N, t_CE_C, t_CE_T, t_CE_V = 0, 0, 0, 0, 0, 0
        b_l_r, b_l_n, t_l_r, t_l_n = 0, 0, 0, 0
        b_l_adv_r, b_l_adv_n, t_l_adv_r, t_l_adv_n = 0, 0, 0, 0
        
        # train on vgmidi
        for j, x in tqdm(enumerate(vgm_train_dl_dist), total=len(vgm_train_dl_dist)):

            d, r, n, c, a, v, r_density, n_density = x
            d, r, n, c = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float()

            d_oh = convert_to_one_hot(d, EVENT_DIMS)
            r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
            n_oh = convert_to_one_hot(n, NOTE_DIMS)

            # c_r_oh = convert_to_one_hot(c_r, 3)
            # c_n_oh = convert_to_one_hot(c_n, 3)

            step, loss = train(step, d_oh, r_oh, n_oh,
                                d, r, n, c, None, None, None, None, r_density, n_density,
                                is_supervised=True, y_label=a)
            loss, CE_X, CE_R, CE_N, l_r, l_n, l_adv_r, l_adv_n = loss
            batch_loss += loss

            inputs = b_CE_X, b_CE_R, b_CE_N
            updates = CE_X, CE_R, CE_N
            outputs = []
            for idx in range(len(inputs)):
                outputs.append(inputs[idx] + updates[idx])
            b_CE_X, b_CE_R, b_CE_N = outputs
            b_l_r += l_r
            b_l_n += l_n
            b_l_adv_r += l_adv_r
            b_l_adv_n += l_adv_n
        
        # evaluate on vgmidi
        for j, x in tqdm(enumerate(vgm_val_dl_dist), total=len(vgm_val_dl_dist)):
            
            d, r, n, c, a, v, r_density, n_density = x
            d, r, n, c = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float()

            d_oh = convert_to_one_hot(d, EVENT_DIMS)
            r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
            n_oh = convert_to_one_hot(n, NOTE_DIMS)

            # c_r_oh = convert_to_one_hot(c_r, 3)
            # c_n_oh = convert_to_one_hot(c_n, 3)

            loss = evaluate(step - 1, d_oh, r_oh, n_oh,
                            d, r, n, c, None, None, None, None, r_density, n_density,
                            is_supervised=True, y_label=a)
            loss, CE_X, CE_R, CE_N, l_r, l_n, l_adv_r, l_adv_n = loss
            batch_test_loss += loss

            inputs = t_CE_X, t_CE_R, t_CE_N
            updates = CE_X, CE_R, CE_N
            outputs = []
            for idx in range(len(inputs)):
                outputs.append(inputs[idx] + updates[idx])
            t_CE_X, t_CE_R, t_CE_N = outputs
            t_l_r += l_r
            t_l_n += l_n
            t_l_adv_r += l_adv_r
            t_l_adv_n += l_adv_n
        
        print('batch loss: {:.5f}  {:.5f}'.format(batch_loss / len(vgm_train_dl_dist),
                                                  batch_test_loss / len(vgm_val_dl_dist)))
        print("Data, Rhythm, Note, Chroma")
        print("train loss by term: {:.5f}  {:.5f}  {:.5f}  {:.5f}  {:.5f}  {:.5f}  {:.5f}".format(
            b_CE_X / len(vgm_train_dl_dist), b_CE_R / len(vgm_train_dl_dist), 
            b_CE_N / len(vgm_train_dl_dist),
            b_l_r / len(vgm_train_dl_dist), b_l_n / len(vgm_train_dl_dist),
            b_l_adv_r / len(vgm_train_dl_dist), b_l_adv_n / len(vgm_train_dl_dist)
        ))
        print("test loss by term: {:.5f}  {:.5f}  {:.5f}  {:.5f}  {:.5f}  {:.5f}  {:.5f}".format(
            t_CE_X / len(vgm_val_dl_dist), t_CE_R / len(vgm_val_dl_dist), 
            t_CE_N / len(vgm_val_dl_dist),
            t_l_r / len(vgm_val_dl_dist), t_l_n / len(vgm_val_dl_dist),
            t_l_adv_r / len(vgm_val_dl_dist), t_l_adv_n / len(vgm_val_dl_dist)
        ))

        print("Saving model...")
        torch.save(model.cpu().state_dict(), save_path)
        model.cuda()

    timestamp = str(datetime.now())
    save_path_timing = 'params/{}.pt'.format(args['name'] + "_" + timestamp)
    torch.save(model.cpu().state_dict(), save_path_timing)

    if torch.cuda.is_available():
        model.cuda()
    print('Model saved as {}!'.format(save_path))


def evaluation_phase():
    print("Evaluate")
    if torch.cuda.is_available():
        model.cuda()

    if os.path.exists(save_path):
        print("Loading {}".format(save_path))
        model.load_state_dict(torch.load(save_path))
    
    def run(dl, is_vgmidi=False):
        
        t_CE_X, t_CE_R, t_CE_N, t_CE_C, t_CE_T, t_CE_V = 0, 0, 0, 0, 0, 0
        t_l_r, t_l_n = 0, 0
        t_l_adv_r, t_l_adv_n = 0, 0
        t_acc_x, t_acc_r, t_acc_n, t_acc_t, t_acc_v = 0, 0, 0, 0, 0
        c_acc_r, c_acc_n, c_acc_t, c_acc_v = 0, 0, 0, 0
        data_len = 0
        linear_r, linear_n = [], []

        for i, x in tqdm(enumerate(dl), total=len(dl)):
            d, r, n, c, a, v, r_density, n_density = x
            d, r, n, c = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float()

            d_oh = convert_to_one_hot(d, EVENT_DIMS)
            r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
            n_oh = convert_to_one_hot(n, NOTE_DIMS)

            res = model(d_oh, r_oh, n_oh, c, None, None, is_class=is_class, is_res=is_res)

            # package output
            output, dis, z_out, logLogit_out, qy_x_out, y_out = res
            out, r_out, n_out, _, _ = output
            z_r, z_n = z_out

            if not is_vgmidi:
                # calculate gmm loss
                loss, CE_X, CE_R, CE_N, kld_lat_r_total, kld_lat_n_total, \
                    kld_cls_r, kld_cls_n = loss_function(out, d,
                                                    r_out, r,
                                                    n_out, n,
                                                    dis,
                                                    qy_x_out,
                                                    logLogit_out,
                                                    step,
                                                    beta=args['beta'])
            
            else:
                # calculate gmm loss
                loss, CE_X, CE_R, CE_N, kld_lat_r_total, kld_lat_n_total, \
                    kld_cls_r, kld_cls_n = loss_function(out, d,
                                                    r_out, r,
                                                    n_out, n,
                                                    dis,
                                                    qy_x_out,
                                                    logLogit_out,
                                                    step,
                                                    beta=args['beta'],
                                                    is_supervised=True,
                                                    y_label=a)
            
            # calculate latent regularization loss
            l_r, l_n = torch.Tensor([0]), torch.Tensor([0])
            l_r, l_n, _ = latent_regularized_loss_function(z_out, r_density, n_density, c=c)

            # adversarial loss
            l_adv_r, l_adv_n = kld_lat_r_total.item() +  kld_lat_n_total.item(), \
                                kld_cls_r.item() + kld_cls_n.item()

            # update
            inputs = t_CE_X, t_CE_R, t_CE_N
            updates = CE_X.item(), CE_R.item(), CE_N.item()
            outputs = []
            for idx in range(len(inputs)):
                outputs.append(inputs[idx] + updates[idx])
            t_CE_X, t_CE_R, t_CE_N = outputs
            t_l_r += l_r.item()
            t_l_n += l_n.item()
            
            # calculate accuracy
            def acc(a, b, t, trim=False):
                a = torch.argmax(a, dim=-1).squeeze().cpu().detach().numpy()
                b = b.squeeze().cpu().detach().numpy()

                b_acc = 0
                for i in range(len(a)):
                    a_batch = a[i]
                    b_batch = b[i]

                    if trim:
                        b_batch = np.trim_zeros(b_batch)
                        a_batch = a_batch[:len(b_batch)]

                    correct = 0
                    for j in range(len(a_batch)):
                        if a_batch[j] == b_batch[j]:
                            correct += 1
                    acc = correct / len(a_batch)
                    b_acc += acc
                
                return b_acc

            acc_x, acc_r, acc_n = acc(out, d, "d", trim=True), \
                                  acc(r_out, r, "r"), acc(n_out, n, "n")
            data_len += out.shape[0]

            # accuracy update store
            inputs = t_acc_x, t_acc_r, t_acc_n
            updates = acc_x, acc_r, acc_n
            outputs = []
            for idx in range(len(inputs)):
                outputs.append(inputs[idx] + updates[idx])
            t_acc_x, t_acc_r, t_acc_n = outputs
        

            from sklearn.linear_model import LinearRegression

            reg = LinearRegression().fit(z_r[:, 0].unsqueeze(-1).cpu().detach().numpy(), 
                                        r_density.unsqueeze(-1).cpu().detach().numpy())
            linear_r.append(reg.score(z_r[:, 0].unsqueeze(-1).cpu().detach().numpy(), 
                    r_density.unsqueeze(-1).cpu().detach().numpy()))

            reg = LinearRegression().fit(z_n[:, 0].unsqueeze(-1).cpu().detach().numpy(), 
                                        n_density.unsqueeze(-1).cpu().detach().numpy())
            linear_n.append(reg.score(z_n[:, 0].unsqueeze(-1).cpu().detach().numpy(), 
                    n_density.unsqueeze(-1).cpu().detach().numpy()))
        
        
        print(sum(linear_r) / len(linear_r), sum(linear_n) / len(linear_n))

        # Print results
        print(data_len)
        print("CE: {:.4}  {:.4}  {:.4}".format(t_CE_X / len(dl),
                                                    t_CE_R / len(dl), 
                                                    t_CE_N / len(dl)))
        
        print("Regularized: {:.4}  {:.4}".format(t_l_r / len(dl),
                                                t_l_n / len(dl)))

        print("Adversarial: {:.4}  {:.4}".format(t_l_adv_r / len(dl),
                                                t_l_adv_n / len(dl)))
        
        print("Acc: {:.4}  {:.4}  {:.4}".format(t_acc_x / data_len,
                                                t_acc_r / data_len, 
                                                t_acc_n / data_len))

    # dl = DataLoader(train_ds_dist, batch_size=128, shuffle=False, num_workers=0)
    # run(dl)
    # dl = DataLoader(test_ds_dist, batch_size=128, shuffle=False, num_workers=0)
    # run(dl)
    dl = DataLoader(vgm_train_ds_dist, batch_size=32, shuffle=False, num_workers=0)
    run(dl)
    dl = DataLoader(vgm_test_ds_dist, batch_size=32, shuffle=False, num_workers=0)
    run(dl)


training_phase(step)
evaluation_phase()

