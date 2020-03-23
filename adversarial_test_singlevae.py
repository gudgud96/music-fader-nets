import json
import torch
from uni_model_2 import *
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
from sklearn.metrics import mutual_info_score, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def convert_to_one_hot(input, dims):
    if type(input) != int and len(input.shape) > 1:
        input_oh = torch.zeros((input.shape[0], input.shape[1], dims)).cuda()
        input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
    else:
        input_oh = torch.zeros((input.shape[0], dims)).cuda()
        input_oh = input_oh.scatter_(-1, input.unsqueeze(-1), 1.)
    return input_oh

def clean_output(out):
    recon = np.trim_zeros(torch.argmax(out, dim=-1).cpu().detach().numpy().squeeze())
    if 1 in recon:
        last_idx = np.argwhere(recon == 1)[0][0]
        recon[recon == 1] = 0
        recon = recon[:last_idx]
    return recon

def repar(mu, stddev, sigma=1):
    eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
    z = mu + stddev * eps  # reparameterization trick
    return z

def rhythm_shift_reg(d, r, n, c, c_r, c_n, target_z_value):
    d_oh = convert_to_one_hot(d, EVENT_DIMS).unsqueeze(0)
    r_oh = convert_to_one_hot(r, RHYTHM_DIMS).unsqueeze(0)
    n_oh = convert_to_one_hot(n, NOTE_DIMS).unsqueeze(0)
    c = c.unsqueeze(0)

    c_r_oh = torch.zeros(3,).cuda().unsqueeze(0)
    c_r_oh[:,c_r] = 1
    c_n_oh = torch.zeros(3,).cuda().unsqueeze(0)
    c_n_oh[:,c_n] = 1
    
    res = model(d_oh, c)
    out, dis, _ = res

    # get original latent variables
    z = repar(dis.mean, dis.stddev)
    z_0 = z[:, 0].item()
    
    # shifting
    z[:, 0] = target_z_value
    model.eval()
    z = torch.cat([z, c], dim=1)  
    
    out = model.global_decoder(z, steps=100)
    return out, z_0


def get_classes(r, n):
    r_density = Counter(r)[1] / len(r)
    if r_density < 0.3: c_r = 0
    elif r_density < 0.5: c_r = 1
    else: c_r = 2

    n_density = sum(n) / len(n)
    if n_density <= 2: c_n = 0
    elif n_density <= 3.5: c_n = 1
    else: c_n = 2
    
    return r_density, n_density, c_r, c_n


def run_through(dl):
    r_mean, n_mean, t_mean, v_mean = [], [], [], []
    r_lst, n_lst = [], []
    c_r_lst, c_n_lst, c_t_lst, c_v_lst = [], [], [], []
    z_lst = []
    r_density_lst, n_density_lst = [], []
    r_density_lst_actual, n_density_lst_actual = [], []
    temp_count = 0

    for j, x in tqdm(enumerate(dl), total=len(dl)):
        d, r, n, c, c_r, c_n, r_density, n_density = x
        d, r, n, c = d.cuda().long(), r.cuda().long(), \
                    n.cuda().long(), c.cuda().float()
        c_r, c_n, = c_r.cuda().long(), c_n.cuda().long()
        
        r_lst.append(r)
        n_lst.append(n)
        r_density_lst.append(r_density.float())
        n_density_lst.append(n_density.float())

        d_oh = convert_to_one_hot(d, EVENT_DIMS)
        r_oh = convert_to_one_hot(r, RHYTHM_DIMS)
        n_oh = convert_to_one_hot(n, NOTE_DIMS)

        c_r_oh = convert_to_one_hot(c_r, 4)
        c_n_oh = convert_to_one_hot(c_n, 4)

        res = model(d_oh, c)

        # package output
        out, dis, z = res
        
        z_lst.append(z.cpu().detach())

    r_density_lst = torch.cat(r_density_lst, dim=0).cpu().detach().numpy()
    n_density_lst = torch.cat(n_density_lst, dim=0).cpu().detach().numpy()
    z_lst = torch.cat(z_lst, dim=0).cpu().detach().numpy()
    r_lst = torch.cat(r_lst, dim=0).cpu().detach().numpy()
    n_lst = torch.cat(n_lst, dim=0).cpu().detach().numpy()

    r_min, r_max = np.amin(z_lst[:, 0]), np.amax(z_lst[:, 0])
    n_min, n_max = np.amin(z_lst[:, 1]), np.amax(z_lst[:, 1])

    return r_density_lst, n_density_lst, z_lst, \
            r_lst, n_lst, \
            r_min, r_max, n_min, n_max


def consistency_metric(density_lst, z_lst, gt_is_one=True):
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


def train_test_evaluation(dl, is_hierachical=False, is_vgmidi=False):

    r_density_lst, n_density_lst, z_lst, \
        r_lst, n_lst, \
        r_min, r_max, n_min, n_max = run_through(dl)

    # consistency
    # print("Rhythm consistency")
    # consistency_metric(r_lst, z_lst, gt_is_one=False)
    # print("Note consistency")
    # consistency_metric(n_lst, z_lst, gt_is_one=True)

    # monotonicity
    r_density_lst = np.expand_dims(np.array(r_density_lst), axis=-1)
    z_r_0_lst = np.expand_dims(z_lst[:, 0], axis=-1)

    reg = LinearRegression().fit(z_r_0_lst, r_density_lst)
    rhythm_linear_score = reg.score(z_r_0_lst, r_density_lst)
    print("Rhythm monotonicity score (z): {}".format(rhythm_linear_score))
    
    n_density_lst = np.expand_dims(np.array(n_density_lst), axis=-1)
    z_n_0_lst = np.expand_dims(z_lst[:, 1], axis=-1)

    reg = LinearRegression().fit(z_n_0_lst, n_density_lst)
    note_linear_score = reg.score(z_n_0_lst, n_density_lst)
    print("Note monotonicity score (z): {}".format(note_linear_score))
    print()

    # get r and n std
    r_std = np.std(r_lst)
    n_std = np.std(n_lst)

    return r_min, r_max, n_min, n_max, r_std, n_std


def rhythm_generation_evaluation(ds, r_min, r_max, r_std, n_std, is_mse=False):
    # run generation, calculate linear regression score
    gap = (r_max - r_min) / 8
    value_lst = np.array([r_min + k * gap for k in range(8)])
    print(r_min, r_max)
    print(value_lst)
    r_density_lst_new = []
    result = []
    i = 0

    r_out_all_lst = []
    n_out_all_lst = []

    r_values_dict = {}

    while len(result) < 100:
        print(len(result), end="\r")
        r_density_lst = []
        n_density_lst = []
        z_r_lst_infer = []
        z_n_lst_infer = []
        z_lst = []

        d, r, n, c, c_r, c_n, r_density, n_density = ds[i]
        d, r, n, c = torch.from_numpy(d).cuda().long(), torch.from_numpy(r).cuda().long(), \
                    torch.from_numpy(n).cuda().long(), torch.from_numpy(c).cuda().float()
        
        r_density_lst.append(r_density)
        n_density_lst.append(n_density)

        d_oh = convert_to_one_hot(d, EVENT_DIMS).unsqueeze(0)
        r_oh = convert_to_one_hot(r, RHYTHM_DIMS).unsqueeze(0)
        n_oh = convert_to_one_hot(n, NOTE_DIMS).unsqueeze(0)

        c_r_oh = None
        c_n_oh = None

        res = model(d_oh, c.unsqueeze(0))
        out, dis, _ = res

        # get original latent variables
        z = repar(dis.mean, dis.stddev)
        z_0 = z[:, 0].item()
        z_lst.append(z)
        
        # generation part
        try:
            r_infer_lst, n_infer_lst = [], []
            for val in value_lst:
                d_shifted, z_r_0 = rhythm_shift_reg(d, r, n, c, c_r, c_n, target_z_value=val)
                pm = magenta_decode_midi(clean_output(d_shifted))
                pm.write('rhythm_temp.mid')

                # get class
                track = pypianoroll.parse('rhythm_temp.mid', beat_resolution=4).tracks
                if len(track) < 1: continue
                pr = track[0].pianoroll
                _, rhythm, note, chroma, _ = get_music_attributes(pr, beat=4)
                r_density_shifted, n_density_shifted, c_r_shifted, c_n_shifted = get_classes(rhythm, note)
                r_density_lst_new.append(r_density_shifted)
                n_infer_lst.append(n_density_shifted)

                # inferred
                z_r_lst_infer.append(z[:, 0].item())
                z_n_lst_infer.append(z[:, 0])

            # consistency, restrictiveness
            r_out_all_lst.append(np.array(r_density_lst_new))
            n_out_all_lst.append(np.array(n_infer_lst))

            # monotonicity
            r_density_lst = np.expand_dims(np.array(r_density_lst_new), axis=-1)
            z_r_0_lst = np.expand_dims(value_lst, axis=-1)
            reg = LinearRegression().fit(z_r_0_lst, r_density_lst)
            result.append(reg.score(z_r_0_lst, r_density_lst))

            for i in range(len(r_density_lst.squeeze())):
                if r_density_lst.squeeze()[i] not in r_values_dict:
                    r_values_dict[r_density_lst.squeeze()[i]] = []
                r_values_dict[r_density_lst.squeeze()[i]].append(z_r_0_lst.squeeze()[i])
        
        except Exception as e:
            print(e)
            print(i)
            i += 1
            r_density_lst_new = []
            continue
        
        i += 1
        r_density_lst_new = []
    
    # consistency
    r_out_all_lst = np.array(r_out_all_lst) / r_std
    n_out_all_lst = np.array(n_out_all_lst) / n_std
    
    consistency_score = np.average(np.std(r_out_all_lst, axis=0))
    variance_score = np.average(np.std(r_out_all_lst, axis=-1))
    restrictiveness_score = np.average(np.std(n_out_all_lst, axis=-1))
    monotonicity_score = sum(result) / len(result)

    # monotonicity
    print("Generator consistency: ", consistency_score)
    print("Generator variance: ", variance_score)
    print("Generator restrictiveness: ", restrictiveness_score)
    print("Generator monotonicity:", monotonicity_score)

    plt.figure(figsize=(8,8))
    x, y = [], []
    for key in r_values_dict:
        values = r_values_dict[key]
        for val in values:
            x.append(key)
            y.append(val)
    

def note_shift_reg(d, r, n, c, c_r, c_n, target_z_value):
    d_oh = convert_to_one_hot(d, EVENT_DIMS).unsqueeze(0)
    r_oh = convert_to_one_hot(r, RHYTHM_DIMS).unsqueeze(0)
    n_oh = convert_to_one_hot(n, NOTE_DIMS).unsqueeze(0)
    c = c.unsqueeze(0)

    c_r_oh = torch.zeros(3,).cuda().unsqueeze(0)
    c_r_oh[:,c_r] = 1
    c_n_oh = torch.zeros(3,).cuda().unsqueeze(0)
    c_n_oh[:,c_n] = 1
    
    res = model(d_oh, c)
    out, dis, _ = res

    # get original latent variables
    z = repar(dis.mean, dis.stddev)
    z_0 = z[:, 0].item()
        
    # shifting
    z[:, 1] = target_z_value
    model.eval()
    z = torch.cat([z, c], dim=1)  
    
    out = model.global_decoder(z, steps=100)
    return out, z_0


def note_generation_evaluation(ds, n_min, n_max, r_std, n_std, is_mse=False):
    # run generation, calculate linear regression score\
    gap = (n_max - n_min) / 8
    value_lst = np.array([n_min + k * gap for k in range(8)])
    print(n_min, n_max)
    print(value_lst)
    n_density_lst_new = []
    result = []
    ds = test_ds_dist
    n_values_dict = {}

    r_out_all_lst = []
    n_out_all_lst = []

    i = 0

    while len(result) < 100:
        r_density_lst = []
        n_density_lst = []
        z_lst = []
        z_r_lst_infer = []
        z_n_lst_infer = []

        print(len(result), end="\r")
        d, r, n, c, c_r, c_n, r_density, n_density = ds[i]
        d, r, n, c = torch.from_numpy(d).cuda().long(), torch.from_numpy(r).cuda().long(), \
                    torch.from_numpy(n).cuda().long(), torch.from_numpy(c).cuda().float()

        r_density_lst.append(r_density)
        n_density_lst.append(n_density)

        d_oh = convert_to_one_hot(d, EVENT_DIMS).unsqueeze(0)
        r_oh = convert_to_one_hot(r, RHYTHM_DIMS).unsqueeze(0)
        n_oh = convert_to_one_hot(n, NOTE_DIMS).unsqueeze(0)

        c_r_oh = None
        c_n_oh = None

        res = model(d_oh, c.unsqueeze(0))
        out, dis, _ = res

        # get original latent variables
        z = repar(dis.mean, dis.stddev)
        z_0 = z[:, 0].item()
        z_lst.append(z)
        

        try:
            r_infer_lst, n_infer_lst = [], []
            for val in value_lst:
                d_shifted, z_n_0 = note_shift_reg(d, r, n, c, c_r, c_n, target_z_value=val)
                pm = magenta_decode_midi(clean_output(d_shifted))
                pm.write('note_temp.mid')

                # get class
                track = pypianoroll.parse('note_temp.mid', beat_resolution=4).tracks
                if len(track) < 1: continue
                pr = track[0].pianoroll
                _, rhythm, note, chroma, _ = get_music_attributes(pr, beat=4)
                r_density_shifted, n_density_shifted, c_r_shifted, c_n_shifted = get_classes(rhythm, note)
                n_density_lst_new.append(n_density_shifted)
                r_infer_lst.append(r_density_shifted)

                # print("z_n: {}  n: {}".format(val, n_density_shifted))

                # inferred
                z_n_lst_infer.append(z[:, 1].item())

            # consistency, restrictiveness
            r_out_all_lst.append(np.array(r_infer_lst))
            n_out_all_lst.append(np.array(n_density_lst_new))

            # monotonicity
            n_density_lst = np.expand_dims(np.array(n_density_lst_new), axis=-1)
            z_n_0_lst = np.expand_dims(value_lst, axis=-1)
            reg = LinearRegression().fit(z_n_0_lst, n_density_lst)
            result.append(reg.score(z_n_0_lst, n_density_lst))

            for i in range(len(n_density_lst.squeeze())):
                if n_density_lst.squeeze()[i] not in n_values_dict:
                    n_values_dict[n_density_lst.squeeze()[i]] = []
                n_values_dict[n_density_lst.squeeze()[i]].append(z_n_0_lst.squeeze()[i])
        
        except Exception as e:
            print(e)
            print(i)
            i += 1
            n_density_lst_new = []
            continue
        
        i += 1
        n_density_lst_new = []
    
    # consistency
    r_out_all_lst = np.array(r_out_all_lst) / r_std     # shape: (#samples, #values)
    n_out_all_lst = np.array(n_out_all_lst) / n_std
    
    consistency_score = np.average(np.std(n_out_all_lst, axis=0))
    variance_score = np.average(np.std(n_out_all_lst, axis=-1))
    restrictiveness_score = np.average(np.std(r_out_all_lst, axis=-1))
    monotonicity_score = sum(result) / len(result)

    # monotonicity
    print("Generator consistency: ", consistency_score)
    print("Generator variance: ", variance_score)
    print("Generator restrictiveness: ", restrictiveness_score)
    print("Generator monotonicity:", sum(result) / len(result))


if __name__ == "__main__":
    # some initialization
    with open('uni_model_config_2.json') as f:
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

    save_path = "params/music_attr_vae_reg_singlevae.pt"
    
    model = MusicAttrSingleVAE(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
                        tempo_dims=TEMPO_DIMS, velocity_dims=VELOCITY_DIMS, chroma_dims=CHROMA_DIMS,
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

    # vgmidi dataloaders
    print("Loading VGMIDI...")
    data_lst, rhythm_lst, note_density_lst, arousal_lst, valence_lst, chroma_lst = get_vgmidi()
    vgm_train_ds_dist = MusicAttrDataset3(data_lst, rhythm_lst, note_density_lst, 
                                    chroma_lst, arousal_lst, valence_lst, mode="train")
    vgm_train_dl_dist = DataLoader(vgm_train_ds_dist, batch_size=32, shuffle=False, num_workers=0)
    vgm_val_ds_dist = MusicAttrDataset3(data_lst, rhythm_lst, note_density_lst, 
                                    chroma_lst, arousal_lst, valence_lst, mode="val")
    vgm_val_dl_dist = DataLoader(vgm_val_ds_dist, batch_size=32, shuffle=False, num_workers=0)
    vgm_test_ds_dist = MusicAttrDataset3(data_lst, rhythm_lst, note_density_lst, 
                                    chroma_lst, arousal_lst, valence_lst, mode="test")
    vgm_test_dl_dist = DataLoader(vgm_test_ds_dist, batch_size=32, shuffle=False, num_workers=0)
    print(len(vgm_train_ds_dist), len(vgm_val_ds_dist), len(vgm_test_ds_dist))
    print()

    # ================= Implementation =================== #
    print("Train")
    _, _, _, _, r_std, n_std = train_test_evaluation(train_dl_dist, is_hierachical=True)
    print("Test")
    r_min, r_max, n_min, n_max, _, _ = train_test_evaluation(test_dl_dist, is_hierachical=True)
    # print("VGMIDI Train")
    # train_test_evaluation(vgm_train_dl_dist, is_hierachical=True, is_vgmidi=True)
    # print("VGMIDI Test")
    # train_test_evaluation(vgm_test_dl_dist, is_hierachical=True, is_vgmidi=True)
    print("Rhythm Generation")
    rhythm_generation_evaluation(test_ds_dist, r_min, r_max, r_std, n_std, is_mse=False)
    print("Note Generation")
    note_generation_evaluation(test_ds_dist, n_min, n_max, r_std, n_std, is_mse=False)