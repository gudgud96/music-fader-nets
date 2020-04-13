import json
import torch
from uni_model_2 import *
import os
from sklearn.model_selection import train_test_split
from ptb_v2 import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pretty_midi
from tqdm import tqdm
from polyphonic_event_based_v2 import *
from collections import Counter
from torch.distributions import Normal
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
sns.set()

# ====================== Constants ======================== #
EVENT_DIMS = 342
RHYTHM_DIMS = 3
NOTE_DIMS = 16
TEMPO_DIMS = 264
VELOCITY_DIMS = 126
CHROMA_DIMS = 24

# ====================== Utility functions ======================== #
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


class BaseEvaluator:
    def __init__(self, ds, epochs=10, num_of_samples=100):
        self.ds = ds
        self.epochs = epochs
        self.num_of_samples = num_of_samples
    
    def evaluate(self, model, min_val, max_val, r_std, n_std):
        #  run generation, calculate linear regression score
        c_lst, v_lst, r_lst, m_lst = [], [], [], []

        for _ in range(self.epochs):
            gap = (max_val - min_val) / 8
            value_lst = np.array([min_val + k * gap for k in range(8)])
            print(min_val, max_val)
            print(value_lst)
            r_density_lst_new, n_density_lst_new = [], []
            result = []
            r_out_all_lst = []
            n_out_all_lst = []
            values_dict = {}
            
            i = 0

            while len(result) < self.num_of_samples:
                print(len(result), end="\r")
                r_density_lst = []
                n_density_lst = []
                z_r_lst_infer = []
                z_n_lst_infer = []
                z_r_lst = []
                z_n_lst = []

                random_idx = random.randint(0, len(self.ds))
                d, r, n, c, c_r, c_n, r_density, n_density = self.ds[random_idx]
                d, r, n, c = torch.from_numpy(d).cuda().long(), torch.from_numpy(r).cuda().long(), \
                            torch.from_numpy(n).cuda().long(), torch.from_numpy(c).cuda().float()
                
                r_density_lst.append(r_density)
                n_density_lst.append(n_density)

                d_oh = convert_to_one_hot(d, EVENT_DIMS).unsqueeze(0)
                r_oh = convert_to_one_hot(r, RHYTHM_DIMS).unsqueeze(0)
                n_oh = convert_to_one_hot(n, NOTE_DIMS).unsqueeze(0)

                c_r_oh = None
                c_n_oh = None

                res = self.model_forward(model, d_oh, r_oh, n_oh, c)

                z_r, z_n = self.handle_z_output(res)
                z_r_lst.append(z_r.cpu().detach())
                z_n_lst.append(z_n.cpu().detach())

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
                n_density_lst_new = []
            
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
        raise NotImplementedError

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

    def handle_z_output(self, res):
        output, dis, z_out = res
        return z_out
    
    def handle_dis_output(self, res):
        output, dis, z_out = res
        return dis


class RhythmEvaluator(BaseEvaluator):
    def __init__(self, ds, epochs=10, num_of_samples=100):
        super().__init__(ds, epochs=epochs, num_of_samples=num_of_samples)
    
    def model_forward(self, model, d_oh, r_oh, n_oh, c):
        return model(d_oh, r_oh, n_oh, c.unsqueeze(0), None, None)
    
    def shift(self, model, d, r, n, c, target_z_value):
        d_oh = convert_to_one_hot(d, EVENT_DIMS).unsqueeze(0)
        r_oh = convert_to_one_hot(r, RHYTHM_DIMS).unsqueeze(0)
        n_oh = convert_to_one_hot(n, NOTE_DIMS).unsqueeze(0)
        
        res = self.model_forward(model, d_oh, r_oh, n_oh, c)        
        z_r, z_n = self.handle_z_output(res)

        # get original latent variables
        dis_r, dis_n = self.handle_dis_output(res)
        z_r = repar(dis_r.mean, dis_r.stddev)
        z_n = repar(dis_n.mean, dis_n.stddev)
        
        z_r_0 = z_r[:, 0].item()
        
        # shifting
        z_r[:, 0] = target_z_value
        model.eval()
        z = torch.cat([z_r, z_n, c.unsqueeze(0)], dim=1)  
        
        out = model.global_decoder(z, steps=100)
        return out, z_r_0
    
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


class NoteEvaluator(BaseEvaluator):
    def __init__(self, ds, epochs=10, num_of_samples=100):
        super().__init__(ds, epochs=epochs, num_of_samples=num_of_samples)
    
    def model_forward(self, model, d_oh, r_oh, n_oh, c):
        return model(d_oh, r_oh, n_oh, c.unsqueeze(0), None, None)
    
    def shift(self, model, d, r, n, c, target_z_value):
        d_oh = convert_to_one_hot(d, EVENT_DIMS).unsqueeze(0)
        r_oh = convert_to_one_hot(r, RHYTHM_DIMS).unsqueeze(0)
        n_oh = convert_to_one_hot(n, NOTE_DIMS).unsqueeze(0)
        
        res = self.model_forward(model, d_oh, r_oh, n_oh, c)
        output, dis, z_out = res
        
        z_r, z_n = z_out

        # get original latent variables
        dis_r, dis_n = dis
        z_r = repar(dis_r.mean, dis_r.stddev)
        z_n = repar(dis_n.mean, dis_n.stddev)
        
        z_n_0 = z_n[:, 0].item()
        
        # shifting
        z_n[:, 0] = target_z_value
        model.eval()
        z = torch.cat([z_r, z_n, c.unsqueeze(0)], dim=1)  
        
        out = model.global_decoder(z, steps=100)
        return out, z_n_0
    
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



