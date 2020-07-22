'''
Controllability evaluation of CVAE and Fader Networks.
'''
import json
import argparse
import torch
from model_v2 import *
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
from test_class import *
import matplotlib.pyplot as plt
import seaborn as sns
import random
sns.set()


class CVAEEvaluator(BaseEvaluator):
    def __init__(self, ds, epochs=10, num_of_samples=100):
        super().__init__(ds, epochs=epochs, num_of_samples=num_of_samples)
    
    def evaluate(self, model):
        #  run generation, calculate linear regression score
        c_lst, v_lst, r_lst, m_lst = [], [], [], []

        for _ in range(self.epochs):
            value_lst = [k / 8 for k in range(1, 9)]
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

                random_idx = random.randint(0, len(self.ds) - 1)
                d, r, n, c, r_density, n_density = self.ds[random_idx]
                d, r, n, c = torch.from_numpy(d).cuda().long(), torch.from_numpy(r).cuda().long(), \
                            torch.from_numpy(n).cuda().long(), torch.from_numpy(c).cuda().float()
                
                r_density_lst.append(r_density)
                n_density_lst.append(n_density)

                d_oh = convert_to_one_hot(d, EVENT_DIMS).unsqueeze(0)
                r_oh = convert_to_one_hot(r, RHYTHM_DIMS).unsqueeze(0)
                n_oh = convert_to_one_hot(n, NOTE_DIMS).unsqueeze(0)

                dis = self.model_forward(model, d_oh, r_density, n_density, c)
                z = repar(dis.mean, dis.stddev)

                try:
                    r_infer_lst, n_infer_lst = [], []
                    for val in value_lst:
                        new_r_density, new_n_density = self.get_values(val, r_density, n_density)
                        
                        z_cur = torch.cat([z, new_r_density, new_n_density], dim=-1)  
                        model.eval()  
                        d_shifted = model.global_decoder(z_cur, steps=100)
                        
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
            r_std = 0.16162585    # pre-calculated from rhythm dataset and note dataset
            n_std = 0.8861338
            r_out_all_lst = np.array(r_out_all_lst) / r_std
            n_out_all_lst = np.array(n_out_all_lst) / n_std
            
            consistency_score = 1 - self.calculate_consistency(r_out_all_lst, n_out_all_lst)
            restrictiveness_score = 1 - self.calculate_restrictiveness(r_out_all_lst, n_out_all_lst)
            monotonicity_score = sum(result) / len(result)

            # monotonicity
            print("Generator consistency: ", consistency_score)
            print("Generator restrictiveness: ", restrictiveness_score)
            print("Generator monotonicity:", monotonicity_score)

            c_lst.append(consistency_score)
            r_lst.append(restrictiveness_score)
            m_lst.append(monotonicity_score)
        
        c_lst = np.array(c_lst)
        r_lst = np.array(r_lst)
        m_lst = np.array(m_lst)

        print("============================================")
        print("Consistency: {} +/- {}".format(np.mean(c_lst), np.std(c_lst)))
        print("Restrictiveness: {} +/- {}".format(np.mean(r_lst), np.std(r_lst)))
        print("Monotonicity: {} +/- {}".format(np.mean(m_lst), np.std(m_lst)))
        print("============================================")
    
    def get_values(self, val, r_density, n_density):
        new_r_density = torch.Tensor([val]).cuda().unsqueeze(-1)
        new_n_density = torch.Tensor([n_density]).cuda().unsqueeze(-1)
        return new_r_density, new_n_density

    def model_forward(self, model, d_oh, r_density, n_density, c):
        dis = model.encoder(d_oh, torch.Tensor([r_density]).cuda().unsqueeze(0), torch.Tensor([n_density]).cuda().unsqueeze(0), c)
        return dis

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


class RhythmCVAEEvaluator(CVAEEvaluator):
    def __init__(self, ds, epochs=10, num_of_samples=100):
        super().__init__(ds, epochs=epochs, num_of_samples=num_of_samples)
    
    def get_values(self, val, r_density, n_density):
        new_r_density = torch.Tensor([val]).cuda().unsqueeze(-1)
        new_n_density = torch.Tensor([n_density]).cuda().unsqueeze(-1)
        return new_r_density, new_n_density

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


class NoteCVAEEvaluator(CVAEEvaluator):
    def __init__(self, ds, epochs=10, num_of_samples=100):
        super().__init__(ds, epochs=epochs, num_of_samples=num_of_samples)
    
    def get_values(self, val, r_density, n_density):
        new_r_density = torch.Tensor([r_density]).cuda().unsqueeze(-1)
        new_n_density = torch.Tensor([val]).cuda().unsqueeze(-1)
        return new_r_density, new_n_density

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


class RhythmFaderNetsEvaluator(RhythmCVAEEvaluator):
    def __init__(self, ds, epochs=10, num_of_samples=100):
        super().__init__(ds, epochs=epochs, num_of_samples=num_of_samples)
    
    def model_forward(self, model, d_oh, r_density, n_density, c):
        dis = model.encoder(d_oh)
        return dis


class NoteFaderNetsEvaluator(NoteCVAEEvaluator):
    def __init__(self, ds, epochs=10, num_of_samples=100):
        super().__init__(ds, epochs=epochs, num_of_samples=num_of_samples)
    
    def model_forward(self, model, d_oh, r_density, n_density, c):
        dis = model.encoder(d_oh)
        return dis


if __name__ == "__main__":
    # determine if running CVAE or Fader Networks
    parser = argparse.ArgumentParser(description='Training CVAE or Fader Networks.')
    parser.add_argument('--is_cvae', action='store_true',
                        help='Evaluating CVAE or Fader Networks')

    input_args = parser.parse_args()
    
    # initialization
    with open('model_config_v2.json') as f:
        args = json.load(f)

    # model dimensions
    EVENT_DIMS = 342
    RHYTHM_DIMS = 3
    NOTE_DIMS = 16
    TEMPO_DIMS = 264
    
    if input_args.is_cvae:
        print("Evaluating CVAE...")
        save_path = "params/music_attr_vae_reg_cvae.pt"
        model = MusicAttrCVAE(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
                        chroma_dims=CHROMA_DIMS,
                        hidden_dims=args['hidden_dim'], z_dims=args['z_dim'], 
                        n_step=args['time_step'])

    else:
        print("Evaluating Fader Networks...")
        save_path = "params/music_attr_fader.pt"
        model = MusicAttrFaderNets(roll_dims=EVENT_DIMS, rhythm_dims=RHYTHM_DIMS, note_dims=NOTE_DIMS, 
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

    # ================= CVAE implementation =================== #
    print("Rhythm Generation")
    if input_args.is_cvae:
        rhythm_evaluator = RhythmCVAEEvaluator(test_ds_dist, epochs=10, num_of_samples=100)
    else:
        rhythm_evaluator = RhythmFaderNetsEvaluator(test_ds_dist, epochs=10, num_of_samples=100)
    rhythm_evaluator.evaluate(model)

    print("Note Generation")
    if input_args.is_cvae:
        note_evaluator = NoteCVAEEvaluator(test_ds_dist, epochs=10, num_of_samples=100)
    else:
        note_evaluator = NoteFaderNetsEvaluator(test_ds_dist, epochs=10, num_of_samples=100)
    note_evaluator.evaluate(model)