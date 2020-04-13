import json
import torch
from gmm_model import *
import os
from sklearn.model_selection import train_test_split
from ptb_v2 import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pretty_midi
from IPython.display import Audio
from tqdm import tqdm
from polyphonic_event_based_v2 import *
from collections import Counter
from torch.distributions import Normal
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import random
from adversarial_test_class import *

sns.set()

class GMMRhythmEvaluator(RhythmEvaluator):
    def __init__(self, ds, epochs=10, num_of_samples=100):
        super().__init__(ds, epochs=epochs, num_of_samples=num_of_samples)
    
    def handle_z_output(self, res):
        output, dis, z_out, logLogit_out, qy_x_out, y_out = res
        return z_out
    
    def handle_dis_output(self, res):
        output, dis, z_out, logLogit_out, qy_x_out, y_out = res
        return dis


class GMMNoteEvaluator(NoteEvaluator):
    def __init__(self, ds, epochs=10, num_of_samples=100):
        super().__init__(ds, epochs=epochs, num_of_samples=num_of_samples)
    
    def handle_z_output(self, res):
        output, dis, z_out, logLogit_out, qy_x_out, y_out = res
        return z_out
    
    def handle_dis_output(self, res):
        output, dis, z_out, logLogit_out, qy_x_out, y_out = res
        return dis


def run_through_gmm(dl):
    r_mean, n_mean, t_mean, v_mean = [], [], [], []
    r_lst, n_lst = [], []
    a_lst = []
    z_r_lst, z_n_lst = [], []
    r_density_lst, n_density_lst = [], []
    temp_count = 0

    for j, x in tqdm(enumerate(dl), total=len(dl)):
        d, r, n, c, a, v, r_density, n_density = x
        d, r, n, c = d.cuda().long(), r.cuda().long(), \
                    n.cuda().long(), c.cuda().float()
        
        r_lst.append(r)
        n_lst.append(n)
        a_lst.append(a)
        r_density_lst.append(r_density.float())
        n_density_lst.append(n_density.float())

        d_oh = convert_to_one_hot(d, EVENT_DIMS)
        r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
        n_oh = convert_to_one_hot(n, NOTE_DIMS)

        res = model(d_oh, r_oh, n_oh, c, None, None)

        # package output
        output, dis, z_out, logLogit_out, qy_x_out, y_out = res
        out, r_out, n_out, _, _ = output
        z_r, z_n = z_out
        dis_r, dis_n = dis
        
        # hierachical part
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
    a_lst = torch.cat(a_lst, dim=0).cpu().detach().numpy()

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
            r_lst, n_lst, a_lst, \
            r_mean, n_mean, \
            z_r_0_lst, z_r_rest_lst, z_n_0_lst, z_n_rest_lst, \
            r_min, r_max, n_min, n_max


    density_dict = {}
    for i in range(len(density_lst)):
        key = str(density_lst[i])
        if key in density_dict:
            density_dict[key].append(i)
        else:
            density_dict[key] = []
            density_dict[key].append(i)
    
    argmin_lst = []

    avg_var = np.zeros(256,)
    count = 0
    for key in density_dict:
        # select z latents with same density value
        idx = density_dict[key]

        if len(idx) >= 3:
            # get the average latent value
            z_mean = np.average(z_lst[idx, :], axis=0)

            var_z = np.zeros(256,)
            for z in z_lst[idx, :]:
                var = np.power(z - z_mean, 2)
                var_z += var
            var_z /= len(z_lst[idx, :]) - 1     # sample variance
            avg_var += var_z

            count += 1

    avg_var /= count                                # average across samples
    avg_var = avg_var / np.std(z_lst, axis=0)       # normalize by std of dataset

    avg_var_idx = np.argsort(avg_var)[:128]         # first 128 dim 
    y_true, y_pred = np.zeros(256,), np.zeros(256,)
    y_pred[avg_var_idx] = 1
    if gt_is_one:                   # use for note density
        y_true[range(128, 256)] = 1 
    else:
        y_true[range(128)] = 1     # ground truth is first 128 dim corresponds to rhythm

    print("F1: ", f1_score(y_true, y_pred))
    print("Precision: ", precision_score(y_true, y_pred))
    print("Recall: ", recall_score(y_true, y_pred))
    print("Acc: ", accuracy_score(y_true, y_pred))
    print()


def train_test_evaluation_gmm(dl):

    r_density_lst, n_density_lst, \
        r_lst, n_lst, a_lst, \
        r_mean, n_mean, \
        z_r_0_lst, z_r_rest_lst, z_n_0_lst, z_n_rest_lst, \
        r_min, r_max, n_min, n_max = run_through_gmm(dl)

    z_r_lst = np.concatenate([np.expand_dims(z_r_0_lst, axis=-1), z_r_rest_lst], axis=-1)
    z_n_lst = np.concatenate([np.expand_dims(z_n_0_lst, axis=-1), z_n_rest_lst], axis=-1)
    z_lst = np.concatenate([z_r_lst, z_n_lst], axis=-1)

    # get r and n std
    r_std = np.std(r_lst)
    n_std = np.std(n_lst)

    return r_min, r_max, n_min, n_max, r_std, n_std


if __name__ == "__main__":
    # some initialization
    with open('gmm_model_config.json') as f:
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
    TEMPO_DIMS = 264
    VELOCITY_DIMS = 126
    CHROMA_DIMS = 24

    is_adversarial = False

    save_path = "params/music_attr_vae_reg_gmm_v3.8.pt"
    
    model = MusicAttrRegGMVAE(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
                        tempo_dims=TEMPO_DIMS, velocity_dims=VELOCITY_DIMS, chroma_dims=CHROMA_DIMS,
                        hidden_dims=args['hidden_dim'], z_dims=args['z_dim'], 
                        n_step=args['time_step'],
                        n_component=2)

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
    train_ds_dist = MusicAttrDataset2(data_lst, rhythm_lst, note_density_lst, 
                                    chroma_lst, mode="train")
    train_dl_dist = DataLoader(train_ds_dist, batch_size=batch_size, shuffle=False, num_workers=0)
    val_ds_dist = MusicAttrDataset2(data_lst, rhythm_lst, note_density_lst, 
                                    chroma_lst, mode="val")
    val_dl_dist = DataLoader(val_ds_dist, batch_size=batch_size, shuffle=False, num_workers=0)
    test_ds_dist = MusicAttrDataset2(data_lst, rhythm_lst, note_density_lst, 
                                    chroma_lst, mode="test")
    test_dl_dist = DataLoader(test_ds_dist, batch_size=batch_size, shuffle=False, num_workers=0)
    dl = test_dl_dist
    print(len(train_ds_dist), len(val_ds_dist), len(test_ds_dist))

    # ================= Normal implementation =================== #
    print("Train")
    _, _, _, _, r_std, n_std = train_test_evaluation_gmm(train_dl_dist)
    print("Test")
    r_min, r_max, n_min, n_max, _, _ = train_test_evaluation_gmm(test_dl_dist)

    rhythm_evaluator = GMMRhythmEvaluator(test_ds_dist)
    note_evaluator = GMMNoteEvaluator(test_ds_dist)
    print("Rhythm Generation")
    rhythm_evaluator.evaluate(model, r_min, r_max, r_std, n_std)
    print("Note Generation")
    note_evaluator.evaluate(model, n_min, n_max, r_std, n_std)