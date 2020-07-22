'''
Controllability evaluation of GLSR.
'''
import json
import torch
from model_v2 import *
import os
from sklearn.model_selection import train_test_split
from ptb_v2 import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
from IPython.display import Audio
from tqdm import tqdm
from polyphonic_event_based_v2 import *
from test_class import *
from collections import Counter
import random


def run_through(dl):
    r_mean, n_mean, t_mean, v_mean = [], [], [], []
    r_lst, n_lst = [], []
    c_r_lst, c_n_lst, c_t_lst, c_v_lst = [], [], [], []
    z_r_lst, z_n_lst = [], []
    r_density_lst, n_density_lst = [], []
    r_density_lst_actual, n_density_lst_actual = [], []
    temp_count = 0

    for j, x in tqdm(enumerate(dl), total=len(dl)):
        d, r, n, c, r_density, n_density = x
        d, r, n, c = d.cuda().long(), r.cuda().long(), \
                    n.cuda().long(), c.cuda().float()
        
        r_lst.append(r)
        n_lst.append(n)
        r_density_lst.append(r_density.float())
        n_density_lst.append(n_density.float())

        d_oh = convert_to_one_hot(d, EVENT_DIMS)
        r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
        n_oh = convert_to_one_hot(n, NOTE_DIMS)

        res = model(d_oh, r_oh, n_oh, c)

        # package output
        output, dis, z_out = res
        
        z_r, z_n = z_out
        dis_r, dis_n = dis
        out, r_out, n_out = output
        
        z_r_lst.append(z_r.cpu().detach())
        z_n_lst.append(z_n.cpu().detach())

        r_mean.append(dis_r.mean.cpu().detach())
        n_mean.append(dis_n.mean.cpu().detach())

    r_mean = torch.cat(r_mean, dim=0).cpu().detach().numpy()
    n_mean = torch.cat(n_mean, dim=0).cpu().detach().numpy()
    r_density_lst = torch.cat(r_density_lst, dim=0).cpu().detach().numpy()
    n_density_lst = torch.cat(n_density_lst, dim=0).cpu().detach().numpy()
    r_lst = torch.cat(r_lst, dim=0).cpu().detach().numpy()
    n_lst = torch.cat(n_lst, dim=0).cpu().detach().numpy()

    z_r_lst = torch.cat(z_r_lst, dim=0).cpu().detach().numpy()
    z_n_lst = torch.cat(z_n_lst, dim=0).cpu().detach().numpy()

    # find value to set at z_r_0
    z_r_0_lst = z_r_lst[:, 0]
    z_r_rest_lst = z_r_lst[:, 1:]
    z_n_0_lst = z_n_lst[:, 0]
    z_n_rest_lst = z_n_lst[:, 1:]

    r_min, r_max = np.amin(z_r_0_lst), np.amax(z_r_0_lst)
    n_min, n_max = np.amin(z_n_0_lst), np.amax(z_n_0_lst)

    return r_density_lst, n_density_lst, \
            r_lst, n_lst, \
            r_mean, n_mean, \
            z_r_0_lst, z_r_rest_lst, z_n_0_lst, z_n_rest_lst, \
            r_min, r_max, n_min, n_max


def train_test_evaluation(dl):

    r_density_lst, n_density_lst, \
        r_lst, n_lst, \
        r_mean, n_mean, \
        z_r_0_lst, z_r_rest_lst, z_n_0_lst, z_n_rest_lst, \
        r_min, r_max, n_min, n_max = run_through(dl)
    
    z_r_lst = np.concatenate([np.expand_dims(z_r_0_lst, axis=-1), z_r_rest_lst], axis=-1)
    z_n_lst = np.concatenate([np.expand_dims(z_n_0_lst, axis=-1), z_n_rest_lst], axis=-1)
    z_lst = np.concatenate([z_r_lst, z_n_lst], axis=-1)

    # get r and n std
    r_std = np.std(r_density_lst.squeeze())
    n_std = np.std(n_density_lst.squeeze())

    return r_min, r_max, n_min, n_max, r_std, n_std



if __name__ == "__main__":
    # some initialization
    with open('model_config_v2.json') as f:
        args = json.load(f)
    if not os.path.isdir('log'):
        os.mkdir('log')
    if not os.path.isdir('params'):
        os.mkdir('params')

    from datetime import datetime
    timestamp = str(datetime.now())
    save_path_timing = 'params/{}.pt'.format(args['name'] + "_" + timestamp)

    # model dimensions
    EVENT_DIMS = 342
    RHYTHM_DIMS = 3
    NOTE_DIMS = 16
    CHROMA_DIMS = 24

    save_path = "params/music_attr_vae_reg_glsr.pt"
    model = MusicAttrRegVAE(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
                        chroma_dims=CHROMA_DIMS,
                        hidden_dims=args['hidden_dim'], z_dims=args['z_dim'], 
                        n_step=args['time_step'])

    if os.path.exists(save_path):
        print("Loading {}".format(save_path))
        model.load_state_dict(torch.load(save_path))
    else:
        print("No save path!!")

    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('CPU mode')

    step, pre_epoch = 0, 0
    batch_size = args["batch_size"]
    # model.train()

    # dataloaders
    data_lst, rhythm_lst, note_density_lst, chroma_lst = get_classic_piano()
    tlen, vlen = int(0.8 * len(data_lst)), int(0.9 * len(data_lst))
    train_ds_dist = YamahaDataset(data_lst, rhythm_lst, note_density_lst, 
                                    chroma_lst, mode="train")
    train_dl_dist = DataLoader(train_ds_dist, batch_size=batch_size, shuffle=False, num_workers=0)
    val_ds_dist = YamahaDataset(data_lst, rhythm_lst, note_density_lst, 
                                    chroma_lst, mode="val")
    val_dl_dist = DataLoader(val_ds_dist, batch_size=batch_size, shuffle=False, num_workers=0)
    test_ds_dist = YamahaDataset(data_lst, rhythm_lst, note_density_lst, 
                                    chroma_lst, mode="test")
    test_dl_dist = DataLoader(test_ds_dist, batch_size=batch_size, shuffle=False, num_workers=0)
    dl = test_dl_dist
    print(len(train_ds_dist), len(val_ds_dist), len(test_ds_dist))

    # ================= Normal implementation =================== #
    print("Train")
    _, _, _, _, r_std, n_std = train_test_evaluation(train_dl_dist)
    print("Test")
    r_min, r_max, n_min, n_max, _, _ = train_test_evaluation(test_dl_dist)

    print("STD: ", r_std, n_std)

    rhythm_evaluator = RhythmEvaluator(test_ds_dist, epochs=2)
    note_evaluator = NoteEvaluator(test_ds_dist, epochs=2)
    print("Rhythm Generation")
    rhythm_evaluator.evaluate(model, r_min, r_max, r_std, n_std)
    print("Note Generation")
    note_evaluator.evaluate(model, n_min, n_max, r_std, n_std)