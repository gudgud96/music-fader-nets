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
import matplotlib.pyplot as plt
import seaborn as sns
import random
from adversarial_test_class import *
sns.set()


class SingleEvaluator(BaseEvaluator):
    def __init__(self, ds, epochs=10, num_of_samples=100):
        super().__init__(ds, epochs=epochs, num_of_samples=num_of_samples)
    
    def evaluate(self, model, min_val, max_val, r_std, n_std):
        c_lst, v_lst, r_lst, m_lst = [], [], [], []
        
        for _ in range(10):
            gap = (max_val - min_val) / 8
            value_lst = np.array([min_val + k * gap for k in range(8)])
            print(min_val, max_val)
            print(value_lst)
            r_density_lst_new, n_density_lst_new = [], []
            result = []
            i = 0

            r_out_all_lst = []
            n_out_all_lst = []

            values_dict = {}

            while len(result) < 100:
                print(len(result), end="\r")
                r_density_lst = []
                n_density_lst = []
                z_r_lst_infer = []
                z_n_lst_infer = []
                z_lst = []

                random_idx = random.randint(0, len(self.ds))
                d, r, n, c, c_r, c_n, r_density, n_density = self.ds[random_idx]
                d, r, n, c = torch.from_numpy(d).cuda().long(), torch.from_numpy(r).cuda().long(), \
                            torch.from_numpy(n).cuda().long(), torch.from_numpy(c).cuda().float()
                
                r_density_lst.append(r_density)
                n_density_lst.append(n_density)

                d_oh = convert_to_one_hot(d, EVENT_DIMS).unsqueeze(0)
                r_oh = convert_to_one_hot(r, RHYTHM_DIMS).unsqueeze(0)
                n_oh = convert_to_one_hot(n, NOTE_DIMS).unsqueeze(0)

                res = self.model_forward(model, d_oh, r_oh, n_oh, c)
                out, dis, _ = res

                # get original latent variables
                z = repar(dis.mean, dis.stddev)
                z_0 = z[:, 0].item()
                z_lst.append(z)
                
                # generation part
                try:
                    r_infer_lst, n_infer_lst = [], []
                    for val in value_lst:
                        d_shifted, z_r_0 = self.shift(model, d, r, n, c, target_z_value=val)
                        pm = magenta_decode_midi(clean_output(d_shifted))
                        pm.write('tmp.mid')

                        # get class
                        track = pypianoroll.parse('tmp.mid', beat_resolution=4).tracks
                        if len(track) < 1: continue
                        pr = track[0].pianoroll
                        _, rhythm, note, chroma, _ = get_music_attributes(pr, beat=4)
                        r_density_shifted, n_density_shifted, _, _ = get_classes(rhythm, note)
                        r_density_lst_new.append(r_density_shifted)
                        n_density_lst_new.append(n_density_shifted)

                        # inferred
                        z_r_lst_infer.append(z[:, 0].item())
                        z_n_lst_infer.append(z[:, 0])
                    
                    if self.is_density_lst_length(r_density_lst_new, n_density_lst_new, value_lst):   
                        # if some tracks has length < 0
                        r_density_lst_new = []
                        n_density_lst_new = []
                        continue

                    # consistency, restrictiveness
                    r_out_all_lst.append(np.array(r_density_lst_new))
                    n_out_all_lst.append(np.array(n_density_lst_new))

                    # monotonicity
                    result.append(self.calculate_monotonicity(r_density_lst_new, 
                                                              n_density_lst_new,
                                                              value_lst))
                
                except Exception as e:
                    print(e)
                    print(i)
                    i += 1
                    r_density_lst_new = []
                    n_density_lst_new = []
                    continue
                
                i += 1
                r_density_lst_new = []
            
            # consistency
            r_out_all_lst = np.array(r_out_all_lst) / r_std
            n_out_all_lst = np.array(n_out_all_lst) / n_std
            
            consistency_score = self.calculate_consistency(r_out_all_lst, n_out_all_lst)
            variance_score = self.calculate_variance(r_out_all_lst, n_out_all_lst)
            restrictiveness_score = self.calculate_restrictiveness(r_out_all_lst, n_out_all_lst)
            monotonicity_score = sum(result) / len(result)

            # monotonicity
            print("Generator consistency: ", consistency_score)
            print("Generator variance: ", variance_score)
            print("Generator restrictiveness: ", restrictiveness_score)
            print("Generator monotonicity:", monotonicity_score)
            c_lst.append(consistency_score)
            v_lst.append(variance_score)
            r_lst.append(restrictiveness_score)
            m_lst.append(monotonicity_score)
        
        c_lst = np.array(c_lst)
        v_lst = np.array(v_lst)
        r_lst = np.array(r_lst)
        m_lst = np.array(m_lst)

        print("============================================")
        print("Consistency: {} +/- {}".format(np.mean(c_lst), np.std(c_lst)))
        print("Variance: {} +/- {}".format(np.mean(v_lst), np.std(v_lst)))
        print("Restrictiveness: {} +/- {}".format(np.mean(r_lst), np.std(r_lst)))
        print("Monotonicity: {} +/- {}".format(np.mean(m_lst), np.std(m_lst)))
        print("============================================")  

    def model_forward(self, model, d_oh, r_oh, n_oh, c):
        res = model(d_oh, c.unsqueeze(0))
        return res

    def shift(self, model, d, r, n, c, target_z_value):
        raise NotImplementedError

    def is_density_lst_length(self, r_density_lst_new, n_density_lst_new, value_lst):
        raise NotImplementedError

    def calculate_consistency(self, r_out_all_lst, n_out_all_lst):
        raise NotImplementedError
    
    def calculate_variance(self, r_out_all_lst, n_out_all_lst):
        raise NotImplementedError
    
    def calculate_restrictiveness(self, r_out_all_lst, n_out_all_lst):
        raise NotImplementedError
    
    def calculate_monotonicity(self, r_density_lst_new, n_density_lst_new, value_lst):
        raise NotImplementedError


class SingleRhythmEvaluator(SingleEvaluator):
    def __init__(self, ds, epochs=10, num_of_samples=100):
        super().__init__(ds, epochs=epochs, num_of_samples=num_of_samples)

    def shift(self, model, d, r, n, c, target_z_value):
        d_oh = convert_to_one_hot(d, EVENT_DIMS).unsqueeze(0)
        r_oh = convert_to_one_hot(r, RHYTHM_DIMS).unsqueeze(0)
        n_oh = convert_to_one_hot(n, NOTE_DIMS).unsqueeze(0)
        c = c.unsqueeze(0)
        
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
    
    def is_density_lst_length(self, r_density_lst_new, n_density_lst_new, value_lst):
        return len(r_density_lst_new) < len(value_lst)

    def calculate_consistency(self, r_out_all_lst, n_out_all_lst):
        return np.average(np.std(r_out_all_lst, axis=0))
    
    def calculate_variance(self, r_out_all_lst, n_out_all_lst):
        return np.average(np.std(r_out_all_lst, axis=-1))
    
    def calculate_restrictiveness(self, r_out_all_lst, n_out_all_lst):
        return np.average(np.std(n_out_all_lst, axis=-1))
    
    def calculate_monotonicity(self, r_density_lst_new, n_density_lst_new, value_lst):
        r_density_lst = np.expand_dims(np.array(r_density_lst_new), axis=-1)
        z_r_0_lst = np.expand_dims(value_lst, axis=-1)
        reg = LinearRegression().fit(z_r_0_lst, r_density_lst)
        return reg.score(z_r_0_lst, r_density_lst)


class SingleNoteEvaluator(SingleEvaluator):
    def __init__(self, ds, epochs=10, num_of_samples=100):
        super().__init__(ds, epochs=epochs, num_of_samples=num_of_samples)

    def shift(self, model, d, r, n, c, target_z_value):
        d_oh = convert_to_one_hot(d, EVENT_DIMS).unsqueeze(0)
        r_oh = convert_to_one_hot(r, RHYTHM_DIMS).unsqueeze(0)
        n_oh = convert_to_one_hot(n, NOTE_DIMS).unsqueeze(0)
        c = c.unsqueeze(0)
        
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
    
    def is_density_lst_length(self, r_density_lst_new, n_density_lst_new, value_lst):
        return len(n_density_lst_new) < len(value_lst)

    def calculate_consistency(self, r_out_all_lst, n_out_all_lst):
        return np.average(np.std(n_out_all_lst, axis=0))
    
    def calculate_variance(self, r_out_all_lst, n_out_all_lst):
        return np.average(np.std(n_out_all_lst, axis=-1))
    
    def calculate_restrictiveness(self, r_out_all_lst, n_out_all_lst):
        return np.average(np.std(r_out_all_lst, axis=-1))
    
    def calculate_monotonicity(self, r_density_lst_new, n_density_lst_new, value_lst):
        n_density_lst = np.expand_dims(np.array(n_density_lst_new), axis=-1)
        z_n_0_lst = np.expand_dims(value_lst, axis=-1)
        reg = LinearRegression().fit(z_n_0_lst, n_density_lst)
        return reg.score(z_n_0_lst, n_density_lst)


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

    # ================= Implementation =================== #
    print("Train")
    _, _, _, _, r_std, n_std = train_test_evaluation(train_dl_dist, is_hierachical=True)
    print("Test")
    r_min, r_max, n_min, n_max, _, _ = train_test_evaluation(test_dl_dist, is_hierachical=True)

    print("Rhythm Generation")
    rhythm_evaluator = SingleRhythmEvaluator(test_ds_dist)
    rhythm_evaluator.evaluate(model, r_min, r_max, r_std, n_std)
    
    print("Note Generation")
    note_evaluator = SingleNoteEvaluator(test_ds_dist)
    note_evaluator.evaluate(model, r_min, r_max, r_std, n_std)
