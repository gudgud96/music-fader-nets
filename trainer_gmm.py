'''
Music FaderNets, GM-VAE model.
'''
import json
import torch
import os
import numpy as np
from gmm_model import *
from torch import optim
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split
from polyphonic_event_based_v2 import parse_pretty_midi
from collections import Counter
from sklearn.metrics import accuracy_score
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
CHROMA_DIMS = 24

model = MusicAttrRegGMVAE(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
                        chroma_dims=CHROMA_DIMS,
                        hidden_dims=args['hidden_dim'], z_dims=args['z_dim'], 
                        n_step=args['time_step'],
                        n_component=args['num_clusters'])

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
train_ds_dist = YamahaDataset(data_lst, rhythm_lst, note_density_lst, 
                                chroma_lst, mode="train")
train_dl_dist = DataLoader(train_ds_dist, batch_size=batch_size, shuffle=is_shuffle, num_workers=0)
val_ds_dist = YamahaDataset(data_lst, rhythm_lst, note_density_lst, 
                                chroma_lst, mode="val")
val_dl_dist = DataLoader(val_ds_dist, batch_size=batch_size, shuffle=is_shuffle, num_workers=0)
test_ds_dist = YamahaDataset(data_lst, rhythm_lst, note_density_lst, 
                                chroma_lst, mode="test")
test_dl_dist = DataLoader(test_ds_dist, batch_size=batch_size, shuffle=is_shuffle, num_workers=0)
dl = train_dl_dist
print("Yamaha: Train / Validation / Test")
print(len(train_ds_dist), len(val_ds_dist), len(test_ds_dist))

# vgmidi dataloaders
print("Loading VGMIDI...")
data_lst, rhythm_lst, note_density_lst, arousal_lst, valence_lst, chroma_lst = get_vgmidi()
vgm_train_ds_dist = VGMIDIDataset(data_lst, rhythm_lst, note_density_lst, 
                                chroma_lst, arousal_lst, valence_lst, mode="train")
vgm_train_dl_dist = DataLoader(vgm_train_ds_dist, batch_size=32, shuffle=is_shuffle, num_workers=0)
vgm_val_ds_dist = VGMIDIDataset(data_lst, rhythm_lst, note_density_lst, 
                                chroma_lst, arousal_lst, valence_lst, mode="val")
vgm_val_dl_dist = DataLoader(vgm_val_ds_dist, batch_size=32, shuffle=is_shuffle, num_workers=0)
vgm_test_ds_dist = VGMIDIDataset(data_lst, rhythm_lst, note_density_lst, 
                                chroma_lst, arousal_lst, valence_lst, mode="test")
vgm_test_dl_dist = DataLoader(vgm_test_ds_dist, batch_size=32, shuffle=is_shuffle, num_workers=0)
print("VGMIDI: Train / Validation / Test")
print(len(vgm_train_ds_dist), len(vgm_val_ds_dist), len(vgm_test_ds_dist))
print()


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
    # anneal beta
    if step < 1000:
        beta0 = 0
    else:
        beta0 = min((step - 10000) / 10000 * beta, beta) 

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

    # Unsupervised loss
    if not is_supervised:
        # KL latent loss
        n_component = qy_x_r.shape[-1]
        
        for k in torch.arange(0, n_component):       # number of components
            # infer current p(z|y)
            mu_pz_y_r, var_pz_y_r = model.mu_r_lookup(k.cuda()), model.logvar_r_lookup(k.cuda()).exp_()
            dis_pz_y_r = Normal(mu_pz_y_r, var_pz_y_r)
            kld_lat_r = torch.mean(kl_divergence(dis_r, dis_pz_y_r), dim=-1)
            kld_lat_r *= qy_x_r[:, k]

            mu_pz_y_n, var_pz_y_n = model.mu_n_lookup(k.cuda()), model.logvar_n_lookup(k.cuda()).exp_()
            dis_pz_y_n = Normal(mu_pz_y_n, var_pz_y_n)
            kld_lat_n = torch.mean(kl_divergence(dis_n, dis_pz_y_n), dim=-1)
            kld_lat_n *= qy_x_n[:, k]

            kld_lat_r_total += kld_lat_r.mean()
            kld_lat_n_total += kld_lat_n.mean()
        
        # KL class loss --> KL[q(y|x) || p(y)] = H(q(y|x)) - log p(y)
        def entropy(qy_x, logLogit_qy_x):
            return torch.mean(qy_x * torch.nn.functional.log_softmax(logLogit_qy_x, dim=1), dim=1)
        
        h_qy_x_r = entropy(qy_x_r, logLogit_qy_x_r)
        h_qy_x_n = entropy(qy_x_n, logLogit_qy_x_n)
        kld_cls_r = (h_qy_x_r - np.log(1 / n_component)).mean()
        kld_cls_n = (h_qy_x_n - np.log(1 / n_component)).mean()

        loss = CE + beta0 * (kld_lat_r_total + kld_lat_n_total + kld_cls_r + kld_cls_n)
    
    # Supervised loss
    else:
        mu_pz_y_r, var_pz_y_r = model.mu_r_lookup(y_label.cuda().long()), model.logvar_r_lookup(y_label.cuda().long()).exp_()
        dis_pz_y_r = Normal(mu_pz_y_r, var_pz_y_r)
        kld_lat_r = torch.mean(kl_divergence(dis_r, dis_pz_y_r), dim=-1)

        mu_pz_y_n, var_pz_y_n = model.mu_n_lookup(y_label.cuda().long()), model.logvar_n_lookup(y_label.cuda().long()).exp_()
        dis_pz_y_n = Normal(mu_pz_y_n, var_pz_y_n)
        kld_lat_n = torch.mean(kl_divergence(dis_n, dis_pz_y_n), dim=-1)

        kld_lat_r_total, kld_lat_n_total = kld_lat_r.mean(), kld_lat_n.mean()

        label_clf_loss = nn.CrossEntropyLoss()(qy_x_r, y_label.cuda().long()) + \
                            nn.CrossEntropyLoss()(qy_x_n, y_label.cuda().long())
        loss = CE + beta0 * (kld_lat_r_total + kld_lat_n_total) + label_clf_loss
        
    return loss, CE_X, CE_R, CE_N, kld_lat_r_total, kld_lat_n_total, kld_cls_r, kld_cls_n


def latent_regularized_loss_function(z_out, r, n):
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

    return l_r, l_n


def train(step, d_oh, r_oh, n_oh, d, r, n, c, r_density, n_density,
          is_supervised=False, y_label=None):
    
    optimizer.zero_grad()
    res = model(d_oh, r_oh, n_oh, c)

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
    l_r, l_n = latent_regularized_loss_function(z_out, r_density, n_density)
    loss += l_r + l_n
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    step += 1

    kld_latent = kld_lat_r_total + kld_lat_n_total
    kld_class = kld_cls_r + kld_cls_n
    
    output = loss.item(), CE_X.item(), CE_R.item(), CE_N.item(), l_r.item(), l_n.item(), kld_latent.item(), kld_class.item()
    return step, output


def evaluate(step, d_oh, r_oh, n_oh, d, r, n, c, r_density, n_density,
             is_supervised=False, y_label=None):
    
    res = model(d_oh, r_oh, n_oh, c)

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
    l_r, l_n = latent_regularized_loss_function(z_out, r_density, n_density)
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
    print("D - Data, R - Rhythm, N - Note, RD - Reg. Rhythm, ND- Reg. Note, KLD-L: KLD Latent, KLD-C: KLD Class")
    for i in range(1, args['n_epochs'] + 1):
        print("Epoch {} / {}".format(i, args['n_epochs']))

        # =================== TRAIN VGMIDI ======================== #

        batch_loss, batch_test_loss = 0, 0
        b_CE_X, b_CE_R, b_CE_N = 0, 0, 0
        t_CE_X, t_CE_R, t_CE_N = 0, 0, 0
        b_l_r, b_l_n, t_l_r, t_l_n = 0, 0, 0, 0
        b_kld_latent, b_kld_class, t_kld_latent, t_kld_class  = 0, 0, 0, 0
        
        # train on vgmidi
        for j, x in tqdm(enumerate(vgm_train_dl_dist), total=len(vgm_train_dl_dist)):

            d, r, n, c, a, v, r_density, n_density = x
            d, r, n, c = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float()

            d_oh = convert_to_one_hot(d, EVENT_DIMS)
            r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
            n_oh = convert_to_one_hot(n, NOTE_DIMS)

            step, loss = train(step, d_oh, r_oh, n_oh,
                                d, r, n, c, r_density, n_density,
                                is_supervised=True, y_label=a)
            loss, CE_X, CE_R, CE_N, l_r, l_n, kld_latent, kld_class = loss
            batch_loss += loss

            b_CE_X += CE_X
            b_CE_R += CE_R
            b_CE_N += CE_N
            b_l_r += l_r
            b_l_n += l_n
            b_kld_latent += kld_latent
            b_kld_class += kld_class
        
        # evaluate on vgmidi
        for j, x in tqdm(enumerate(vgm_val_dl_dist), total=len(vgm_val_dl_dist)):
            
            d, r, n, c, a, v, r_density, n_density = x
            d, r, n, c = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float()

            d_oh = convert_to_one_hot(d, EVENT_DIMS)
            r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
            n_oh = convert_to_one_hot(n, NOTE_DIMS)

            loss = evaluate(step - 1, d_oh, r_oh, n_oh,
                            d, r, n, c, r_density, n_density,
                            is_supervised=True, y_label=a)
            loss, CE_X, CE_R, CE_N, l_r, l_n, kld_latent, kld_class = loss
            batch_test_loss += loss
            
            t_CE_X += CE_X
            t_CE_R += CE_R
            t_CE_N += CE_N
            t_l_r += l_r
            t_l_n += l_n
            t_kld_latent += kld_latent
            t_kld_class += kld_class
        
        print('batch loss: {:.5f}  {:.5f}'.format(batch_loss / len(vgm_train_dl_dist),
                                                  batch_test_loss / len(vgm_val_dl_dist)))

        print("train loss by term - D: {:.4f} R: {:.4f} N: {:.4f} RD: {:.4f} ND: {:.4f} KLD-L: {:.4f} KLD-C: {:.4f}".format(
            b_CE_X / len(vgm_train_dl_dist), b_CE_R / len(vgm_train_dl_dist), 
            b_CE_N / len(vgm_train_dl_dist),
            b_l_r / len(vgm_train_dl_dist), b_l_n / len(vgm_train_dl_dist),
            b_kld_latent / len(vgm_train_dl_dist), b_kld_class / len(vgm_train_dl_dist)
        ))
        print("test loss by term - D: {:.4f} R: {:.4f} N: {:.4f} RD: {:.4f} ND: {:.4f} KLD-L: {:.4f} KLD-C: {:.4f}".format(
            t_CE_X / len(vgm_val_dl_dist), t_CE_R / len(vgm_val_dl_dist), 
            t_CE_N / len(vgm_val_dl_dist),
            t_l_r / len(vgm_val_dl_dist), t_l_n / len(vgm_val_dl_dist),
            t_kld_latent / len(vgm_val_dl_dist), t_kld_class / len(vgm_val_dl_dist)
        ))

        # =================== TRAIN YAMAHA ======================== #
        
        batch_loss, batch_test_loss = 0, 0
        b_CE_X, b_CE_R, b_CE_N = 0, 0, 0
        t_CE_X, t_CE_R, t_CE_N = 0, 0, 0
        b_l_r, b_l_n, t_l_r, t_l_n = 0, 0, 0, 0
        b_kld_latent, b_kld_class, t_kld_latent, t_kld_class  = 0, 0, 0, 0

        # train on yamaha
        for j, x in tqdm(enumerate(train_dl_dist), total=len(train_dl_dist)):

            d, r, n, c, r_density, n_density = x
            d, r, n, c = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float()

            d_oh = convert_to_one_hot(d, EVENT_DIMS)
            r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
            n_oh = convert_to_one_hot(n, NOTE_DIMS)

            step, loss = train(step, d_oh, r_oh, n_oh, d, r, n, c, 
                               r_density, n_density)
            loss, CE_X, CE_R, CE_N, l_r, l_n, kld_latent, kld_class = loss
            batch_loss += loss

            b_CE_X += CE_X
            b_CE_R += CE_R
            b_CE_N += CE_N
            b_l_r += l_r
            b_l_n += l_n
            b_kld_latent += kld_latent
            b_kld_class += kld_class

        # evaluate on yamaha
        for j, x in tqdm(enumerate(val_dl_dist), total=len(val_dl_dist)):
            
            d, r, n, c, r_density, n_density = x
            d, r, n, c = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float()

            d_oh = convert_to_one_hot(d, EVENT_DIMS)
            r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
            n_oh = convert_to_one_hot(n, NOTE_DIMS)

            loss = evaluate(step - 1, d_oh, r_oh, n_oh, d, r, n, c, 
                            r_density, n_density)

            loss, CE_X, CE_R, CE_N, l_r, l_n, kld_latent, kld_class = loss
            batch_test_loss += loss

            t_CE_X += CE_X
            t_CE_R += CE_R
            t_CE_N += CE_N
            t_l_r += l_r
            t_l_n += l_n
            t_kld_latent += kld_latent
            t_kld_class += kld_class
        
        print('batch loss: {:.5f}  {:.5f}'.format(batch_loss / len(train_dl_dist),
                                                  batch_test_loss / len(val_dl_dist)))
        print("train loss by term - D: {:.4f} R: {:.4f} N: {:.4f} RD: {:.4f} ND: {:.4f} KLD-L: {:.4f} KLD-C: {:.4f}".format(
            b_CE_X / len(train_dl_dist), b_CE_R / len(train_dl_dist), 
            b_CE_N / len(train_dl_dist),
            b_l_r / len(train_dl_dist), b_l_n / len(train_dl_dist),
            b_kld_latent / len(train_dl_dist), b_kld_class / len(train_dl_dist)
        ))
        print("test loss by term - D: {:.4f} R: {:.4f} N: {:.4f} RD: {:.4f} ND: {:.4f} KLD-L: {:.4f} KLD-C: {:.4f}".format(
            t_CE_X / len(val_dl_dist), t_CE_R / len(val_dl_dist), 
            t_CE_N / len(val_dl_dist),
            t_l_r / len(val_dl_dist), t_l_n / len(val_dl_dist),
            t_kld_latent / len(val_dl_dist), t_kld_class / len(val_dl_dist)
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
        
        t_CE_X, t_CE_R, t_CE_N = 0, 0, 0
        t_l_r, t_l_n = 0, 0
        t_kld_latent, t_kld_class = 0, 0
        t_acc_x, t_acc_r, t_acc_n, t_acc_a_r, t_acc_a_n = 0, 0, 0, 0, 0
        data_len = 0

        for i, x in tqdm(enumerate(dl), total=len(dl)):
            d, r, n, c, a, v, r_density, n_density = x
            d, r, n, c = d.cuda().long(), r.cuda().long(), \
                         n.cuda().long(), c.cuda().float()

            d_oh = convert_to_one_hot(d, EVENT_DIMS)
            r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
            n_oh = convert_to_one_hot(n, NOTE_DIMS)

            res = model(d_oh, r_oh, n_oh, c)

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
            l_r, l_n = latent_regularized_loss_function(z_out, r_density, n_density)

            # adversarial loss
            kld_latent, kld_class = kld_lat_r_total.item() +  kld_lat_n_total.item(), \
                                    kld_cls_r.item() + kld_cls_n.item()
            
            t_CE_X += CE_X
            t_CE_R += CE_R
            t_CE_N += CE_N
            t_l_r += l_r.item()
            t_l_n += l_n.item()
            t_kld_latent += kld_latent
            t_kld_class += kld_class
            
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

            if is_vgmidi:
                qy_x_r, qy_x_n = qy_x_out
                qy_x_r, qy_x_n = torch.argmax(qy_x_r, axis=-1).cpu().detach().numpy(), \
                                torch.argmax(qy_x_n, axis=-1).cpu().detach().numpy()
                acc_q_x_r = accuracy_score(a.cpu().detach().numpy(), qy_x_r)
                acc_q_x_n = accuracy_score(a.cpu().detach().numpy(), qy_x_n)
            else:
                acc_q_x_r, acc_q_x_n = 0, 0

            t_acc_x += acc_x
            t_acc_r += acc_r
            t_acc_n += acc_n
            t_acc_a_r += acc_q_x_r
            t_acc_a_n += acc_q_x_n

        # Print results
        print("CE: {:.4}  {:.4}  {:.4}".format(t_CE_X / len(dl),
                                                    t_CE_R / len(dl), 
                                                    t_CE_N / len(dl)))
        
        print("Regularized: {:.4}  {:.4}".format(t_l_r / len(dl),
                                                t_l_n / len(dl)))

        print("Adversarial: {:.4}  {:.4}".format(t_l_adv_r / len(dl),
                                                t_l_adv_n / len(dl)))
        
        print("Acc: {:.4}  {:.4}  {:.4}  {:.4}  {:.4}".format(t_acc_x / data_len,
                                                            t_acc_r / data_len, 
                                                            t_acc_n / data_len,
                                                            t_acc_a_r / data_len,
                                                            t_acc_a_n / data_len))

    dl = DataLoader(train_ds_dist, batch_size=128, shuffle=False, num_workers=0)
    run(dl)
    dl = DataLoader(test_ds_dist, batch_size=128, shuffle=False, num_workers=0)
    run(dl)
    dl = DataLoader(vgm_train_ds_dist, batch_size=32, shuffle=False, num_workers=0)
    run(dl, is_vgmidi=True)
    dl = DataLoader(vgm_test_ds_dist, batch_size=32, shuffle=False, num_workers=0)
    run(dl, is_vgmidi=True)


training_phase(step)
evaluation_phase()

