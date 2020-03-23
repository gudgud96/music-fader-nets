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



def repar(mu, stddev, sigma=1):
    eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
    z = mu + stddev * eps  # reparameterization trick
    return z


def rhythm_generation_evaluation_cvae(ds, is_cvae=True):
    # run generation, calculate linear regression score
    value_lst = [k / 8 for k in range(1, 9)]
    print(value_lst)
    r_density_lst_new = []
    result = []
    i = 0

    r_out_all_lst = []
    n_out_all_lst = []

    while len(result) < 100:
        print(len(result), end="\r")
        r_density_lst = []
        n_density_lst = []
        z_r_lst = []
        z_n_lst = []

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

        if is_cvae:
            dis = model.encoder(d_oh, torch.Tensor([r_density]).cuda().unsqueeze(0), torch.Tensor([n_density]).cuda().unsqueeze(0), c)
        else:
            dis = model.encoder(d_oh)


        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z = repar(dis.mean, dis.stddev)
        
        try:
            r_infer_lst, n_infer_lst = [], []
            for val in value_lst:
                new_r_density = torch.Tensor([val]).cuda().unsqueeze(-1)
                new_n_density = torch.Tensor([n_density]).cuda().unsqueeze(-1)
                z_cur = torch.cat([z, new_r_density, new_n_density], dim=-1)  
                model.eval()  
                d_shifted = model.global_decoder(z_cur, steps=100)
                
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

                # print("z_r: {}  r: {}".format(val, r_density_shifted))
            
            # consistency, restrictiveness
            r_out_all_lst.append(np.array(r_density_lst_new))
            n_out_all_lst.append(np.array(n_infer_lst))
        
            # evaluate
            r_density_lst = np.expand_dims(np.array(r_density_lst_new), axis=-1)
            z_r_0_lst = np.expand_dims(value_lst, axis=-1)
            reg = LinearRegression().fit(z_r_0_lst, r_density_lst)
            result.append(reg.score(z_r_0_lst, r_density_lst))
            
        except Exception as e:
            print(e)
            print(i)
            i += 1
            r_density_lst_new = []
            continue
        
        i += 1
        r_density_lst_new = []
    
    # consistency
    r_std = 1
    n_std = 1
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
    

def note_generation_evaluation_cvae(ds, is_cvae=True):
    # run generation, calculate linear regression score
    n_min, n_max = 1, 9
    gap = (n_max - n_min) / 8
    value_lst = np.array([n_min + k * gap for k in range(8)])
    print(value_lst)
    n_density_lst_new = []
    result = []
    i = 0

    r_out_all_lst = []
    n_out_all_lst = []

    while len(result) < 100:
        print(len(result), end="\r")
        r_density_lst = []
        n_density_lst = []
        z_r_lst = []
        z_n_lst = []

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

        if is_cvae:
            dis = model.encoder(d_oh, torch.Tensor([r_density]).cuda().unsqueeze(0), torch.Tensor([n_density]).cuda().unsqueeze(0), c)
        else:
            dis = model.encoder(d_oh)   

        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z

        z = repar(dis.mean, dis.stddev)
        
        try:
            r_infer_lst, n_infer_lst = [], []
            for val in value_lst:
                new_r_density = torch.Tensor([r_density]).cuda().unsqueeze(-1)
                new_n_density = torch.Tensor([val]).cuda().unsqueeze(-1)
                z_cur = torch.cat([z, new_r_density, new_n_density], dim=-1)   
                model.eval()  
                d_shifted = model.global_decoder(z_cur, steps=100)
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
            
            # consistency, restrictiveness
            r_out_all_lst.append(np.array(n_density_lst_new))
            n_out_all_lst.append(np.array(r_infer_lst))

            # evaluate
            n_density_lst = np.expand_dims(np.array(n_density_lst_new), axis=-1)
            z_n_0_lst = np.expand_dims(value_lst, axis=-1)
            reg = LinearRegression().fit(z_n_0_lst, n_density_lst)
            result.append(reg.score(z_n_0_lst, n_density_lst))
        
        except Exception as e:
            print(e)
            print(i)
            i += 1
            n_density_lst_new = []
            continue
        
        i += 1
        n_density_lst_new = []
    
    # consistency
    r_std = 1
    n_std = 1
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

    save_path = "params/music_attr_fader.pt"
    
    # model = MusicAttrCVAE(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
    #                     tempo_dims=TEMPO_DIMS, velocity_dims=VELOCITY_DIMS, chroma_dims=CHROMA_DIMS,
    #                     hidden_dims=args['hidden_dim'], z_dims=args['z_dim'], 
    #                     n_step=args['time_step'])
    model = MusicAttrFaderNets(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
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

    # ================= CVAE implementation =================== #
    print("Rhythm Generation")
    rhythm_generation_evaluation_cvae(test_ds_dist, is_cvae=False)
    print("Note Generation")
    note_generation_evaluation_cvae(test_ds_dist, is_cvae=False)