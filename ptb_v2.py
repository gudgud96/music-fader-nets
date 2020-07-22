import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
# from tslearn.clustering import TimeSeriesKMeans
from tqdm import tqdm
import pretty_midi
from collections import Counter
import sys, math
import pypianoroll
from polyphonic_event_based_v2 import *
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.preprocessing import StandardScaler
import music21

# custom magenta
import magenta
from magenta.models.score2perf.music_encoders import MidiPerformanceEncoder


# define constants
PR_TIME_STEPS = 64
NUM_VELOCITY_BINS = 64
STEPS_PER_SECOND = 100
MIN_PITCH = 21
MAX_PITCH = 108
MIN_NOTE_DENSITY = 0
MAX_NOTE_DENSITY = 13
MIN_TEMPO = 57
MAX_TEMPO = 258
MIN_VELOCITY = 0
MAX_VELOCITY = 126


def magenta_encode_midi(midi_filename, is_eos=False):
    mpe = MidiPerformanceEncoder(
            steps_per_second=STEPS_PER_SECOND,
            num_velocity_bins=NUM_VELOCITY_BINS,
            min_pitch=MIN_PITCH,
            max_pitch=MAX_PITCH,
            add_eos=is_eos)
    ns = magenta.music.midi_file_to_sequence_proto(midi_filename)
    return mpe.encode_note_sequence(ns)


def magenta_decode_midi(notes, is_eos=False):
    mpe = MidiPerformanceEncoder(
            steps_per_second=STEPS_PER_SECOND,
            num_velocity_bins=NUM_VELOCITY_BINS,
            min_pitch=MIN_PITCH,
            max_pitch=MAX_PITCH,
            add_eos=is_eos)
    pm = mpe.decode(notes, return_pm=True)
    return pm


def slice_midi(pm, beats, start_idx, end_idx):
    '''
    Slice given pretty_midi object into number of beat segments.
    '''
    new_pm = pretty_midi.PrettyMIDI()
    new_inst = pretty_midi.Instrument(program=pm.instruments[0].program,
                                      is_drum=pm.instruments[0].is_drum,
                                      name=pm.instruments[0].name)
    start, end = beats[start_idx], beats[end_idx]
    for i in range(len(pm.instruments)):
        for note in pm.instruments[i].notes:
            velocity, pitch = note.velocity, note.pitch
            if note.start > end or note.start < start:
                continue
            else:
                s = note.start - start
                if note.end > end:
                    e = end - start
                else:
                    e = note.end - start
            new_note = pretty_midi.Note(
                velocity=velocity, pitch=pitch, start=s, end=e)
            new_inst.notes.append(new_note)

        for ctrl in pm.instruments[i].control_changes:
            if ctrl.time >= start and ctrl.time < end:
                new_ctrl = pretty_midi.ControlChange(
                    number=ctrl.number, value=ctrl.value, time=ctrl.time - start)
                new_inst.control_changes.append(new_ctrl)

    new_pm.instruments.append(new_inst)
    new_pm.write('tmp.mid')
    return new_pm


def get_harmony_vector(fname, is_one_hot=False):
    '''
    Obtain estimated key for a given music segment with music21 library.
    '''
    CHORD_DICT = {
    "C-": 11, "C": 0, "C#": 1, "D-": 1, "D": 2, "D#": 3, "E-": 3, "E": 4, "E#": 5,
    "F-": 4, "F": 5, "F#": 6, "G-": 6, "G": 7, "G#": 8, "A-": 8, "A": 9, "A#": 10, 
    "B-": 10, "B": 11, "B#": 0
    }

    try:
        score = music21.converter.parse(fname)
        key = score.analyze('key')
        res = np.zeros(24,)
        name, mode = key.tonic.name, key.mode
        idx = CHORD_DICT[name] + 12 if mode == "minor" else CHORD_DICT[name]

        if not is_one_hot:    # output probability of each mode instead of one-hot
            res[idx] = key.correlationCoefficient
            for i, x in enumerate(key.alternateInterpretations):
                name, mode = x.tonic.name, x.mode
                idx = CHORD_DICT[name] + 12 if mode == "minor" else CHORD_DICT[name]
                res[idx] = x.correlationCoefficient

            # zero out negative values
            res[res < 0.1] = 0
        
        else:
            res[idx] = 1

        return res

    except Exception as e:
        print(e, "harmony vector")
        return None


def get_music_attributes(pr, beat=24):
    '''
    Get musical attributes including rhythm density, note_density, chroma and velocity
    for a given piano roll segment.
    '''
    events, pitch_lst, velocity_lst, pr, rhythm = encode_midi(pr, beat=beat, is_pr=True)

    # get note density
    note_density = np.array([len(k) for k in pitch_lst])

    # get chroma
    chroma = np.zeros((pr.shape[0], 12))
    for note in range(12):
        chroma[:, note] = np.sum(pr[:, note::12], axis=1)
    
    # get velocity
    velocity = []
    for i in range(len(pr)):
        if len(np.nonzero(pr[i])[0]) > 0:
            velocity.append(int(np.sum(pr[i]) / len(np.nonzero(pr[i])[0])))
        else:
            velocity.append(0)
    velocity = np.array(velocity)

    return events, rhythm, note_density, chroma, velocity


def get_average_av_values(av_dict, key):
    '''
    Obtain average arousal and valence values from annotation dictionary.
    '''
    arousal_values = []
    valence_values = []
    for i in range(1, 31):
        new_key = "{}_{}".format(key, i)
        if new_key in av_dict and av_dict[new_key]["musicianship"] >= 3:
            arousal_values.append(av_dict[new_key]["arousal"])
            valence_values.append(av_dict[new_key]["valence"])
        else:
            pass

    arousal_values = np.array(arousal_values)
    valence_values = np.array(valence_values)
    
    # filtering algorithm according to Ferreira et al.
    clusters = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=0).fit_predict(arousal_values)

    c1, c2,  c3 = [], [], []
    
    for j in range(len(clusters)):
        if clusters[j] == 0:
            c1.append(arousal_values[j])
        elif clusters[j] == 1:
            c2.append(arousal_values[j])
        elif clusters[j] == 2:
            c3.append(arousal_values[j])
    
    var1 = np.mean(np.var(c1, axis=0))
    var2 = np.mean(np.var(c2, axis=0))
    var3 = np.mean(np.var(c3, axis=0))
    min_var = min(min(var1, var2), var3)
    
    if var1 >= var2 and var1 >= var3:
        if len(c2) > len(c3):
            arousal_values = c2
        else:
            arousal_values = c3
    elif var2 >= var1 and var2 >= var3:
        if len(c1) > len(c3):
            arousal_values = c1
        else:
            arousal_values = c3
    elif var3 >= var2 and var3 >= var1:
        if len(c2) > len(c1):
            arousal_values = c2
        else:
            arousal_values = c1
    
    # aggregate mean for extracted values
    arousal_values = np.mean(arousal_values, axis=0)
    valence_values = np.mean(valence_values, axis=0)

    return arousal_values, valence_values


def process_data(name, beat_res=4, num_of_beats=4, max_tokens=100):
    '''
    Utility function for each data function to extract required data.
    '''
    data_lst = []
    rhythm_lst = []
    note_density_lst = []
    chroma_lst = []
    
    track = pypianoroll.parse(name, beat_resolution=beat_res).tracks

    if len(track) > 0:
        try:
            pm = pretty_midi.PrettyMIDI(name)
            beats = pm.get_beats()
            tempo = pm.get_tempo_changes()
            cur_idx, tempo_new = 0, []
        
        except Exception as e:
            print(e)

        pr = track[0].pianoroll

        # extract segment by segment
        for j in range(0, len(pr), beat_res * num_of_beats):
            start_idx = j
            end_idx = j + beat_res * num_of_beats

            if end_idx // beat_res < len(beats):
                new_pr = pr[start_idx : end_idx]
                new_pm = slice_midi(pm, beats, start_idx // beat_res, end_idx // beat_res)
                new_pm.write("tmp.mid")
                ms = np.argmax(new_pr, axis=-1)

                # ensure each segment is not empty and contain unique notes
                if len(new_pm.instruments[0].notes) > 0 and \
                    len(np.unique(ms)) > 2 and np.count_nonzero(ms) >= 0.75 * len(ms):

                    # get musical attributes
                    _, rhythm, note_density, chroma, \
                        velocity = get_music_attributes(new_pr, beat=beat_res)

                    # get midi encoding sequence
                    events = magenta_encode_midi("tmp.mid")
                    events.append(1)    # EOS token

                    # filter out segments that start with 0 and limit token length
                    if rhythm[0] == 1 and len(events) <= max_tokens:   
                        chroma = get_harmony_vector()    # read from saved "tmp.mid" file
                        
                        # aggregate data points
                        data_lst.append(torch.Tensor(events))
                        rhythm_lst.append(rhythm)
                        note_density_lst.append(note_density)
                        chroma_lst.append(chroma)
    
    return data_lst, rhythm_lst, note_density_lst, chroma_lst


def get_classic_piano(data_type="short"):
    '''
    Main data function for Yamaha Piano e-Competition dataset.
    '''
    labelled_midi = ["/data/haohao_tan/haohao/classic-piano/" + k \
                      for k in os.listdir("/data/haohao_tan/haohao/classic-piano/")]
    labelled_midi += ["/data/haohao_tan/haohao/piano-e-competition/" + k \
                      for k in os.listdir("/data/haohao_tan/haohao/piano-e-competition/")]

    print("Dataset length:", len(labelled_midi))
    keylst = labelled_midi

    if not os.path.exists("data/values_v3/data.npy"):
        data_lst = []
        rhythm_lst = []
        note_density_lst = []
        tempo_change_lst = []
        velocity_lst = []
        chroma_lst = []
        key_signature_lst = []

        for i, name in tqdm(enumerate(keylst), total=len(keylst)):
            try:
                # process data
                if data_type == "short":
                    beat_res, num_of_beats, max_tokens = 4, 4, 100
                elif data_type == "long":
                    beat_res, num_of_beats, max_tokens = 4, 16, 250
                    
                cur_data_lst, cur_rhythm_lst, cur_note_lst, cur_chroma_lst = process_data(name,
                                                                                          beat_res=beat_res, 
                                                                                          num_of_beats=num_of_beats, 
                                                                                          max_tokens=max_tokens)
                data_lst += cur_data_lst
                rhythm_lst += cur_rhythm_lst
                note_density_lst += cur_note_lst
                chroma_lst += cur_chroma_lst
        
            except Exception as e:
                print(e)
                print("Current dataset: {}".format(len(data_lst)))

        # consolidate data
        data_lst = torch.nn.utils.rnn.pad_sequence(data_lst, batch_first=True).numpy().astype(int)
        rhythm_lst = np.array(rhythm_lst)
        note_density_lst = np.array(note_density_lst)
        chroma_lst = np.array(chroma_lst)

        # shuffle data
        np.random.seed(777)
        idx = np.arange(len(data_lst))
        np.random.shuffle(idx)
        data_lst, rhythm_lst, note_density_lst, chroma_lst = data_lst[idx], \
                                                            rhythm_lst[idx], \
                                                            note_density_lst[idx], \
                                                            chroma_lst[idx]

        print("Shapes for: Data, Rhythm Density, Note Density, Chroma")
        print(data_lst.shape, rhythm_lst.shape, note_density_lst.shape, chroma_lst.shape)

        np.save("data/values_v3/data.npy", data_lst)
        np.save("data/values_v3/rhythm.npy", rhythm_lst)
        np.save("data/values_v3/note_density.npy", note_density_lst)
        np.save("data/values_v3/chroma.npy", chroma_lst)

        print("Dataset saved!")
    
    else:
        data_lst = np.load("data/values_v3/data.npy")
        rhythm_lst = np.load("data/values_v3/rhythm.npy")
        note_density_lst = np.load("data/values_v3/note_density.npy")
        chroma_lst = np.load("data/values_v3/chroma.npy")

        # sanitization
        idx = []

        for i in tqdm(range(len(chroma_lst))):
            c = chroma_lst[i]
            third_largest = -np.sort(-c)[2]
            c[c < third_largest] = 0
            chroma_lst[i] = c
            if np.count_nonzero(chroma_lst[i]) == 0:
                idx.append(i)

        data_lst = np.delete(data_lst, idx, axis=0)
        rhythm_lst = np.delete(rhythm_lst, idx, axis=0)
        note_density_lst = np.delete(note_density_lst, idx, axis=0)
        chroma_lst = np.delete(chroma_lst, idx, axis=0)

        print("Shapes for: Data, Rhythm Density, Note Density, Chroma")
        print(data_lst.shape, rhythm_lst.shape, note_density_lst.shape, chroma_lst.shape)

    return data_lst, rhythm_lst, note_density_lst, chroma_lst


def get_vgmidi():
    '''
    Main data function for VGMIDI dataset.
    '''
    data_lst = np.load("data/filtered_songs_disambiguate/song_tokens.npy", allow_pickle=True)
    rhythm_lst = np.load("data/filtered_songs_disambiguate/rhythm_lst.npy", allow_pickle=True)
    note_density_lst = np.load("data/filtered_songs_disambiguate/note_lst.npy", allow_pickle=True)
    valence_lst = np.load("data/filtered_songs_disambiguate/valence_lst.npy")
    arousal_lst = np.load("data/filtered_songs_disambiguate/arousal_lst.npy")

    if os.path.exists("data/filtered_songs_disambiguate/chroma_lst.npy"):
        chroma_lst = np.load("data/filtered_songs_disambiguate/chroma_lst.npy")
    else:
        chroma_lst = []
        for _, token in tqdm(enumerate(data_lst), total=len(data_lst)):
            pm = magenta_decode_midi(token)
            pm.write("vgmidi_tmp.mid")
            chroma = get_harmony_vector("vgmidi_tmp.mid", is_one_hot=True)
            chroma_lst.append(chroma)
        chroma_lst = np.array(chroma_lst)
        np.save("data/filtered_songs_disambiguate/chroma_lst.npy", chroma_lst)
    
    print("Shapes for: Data, Rhythm Density, Note Density, Chroma")
    print(data_lst.shape, rhythm_lst.shape, note_density_lst.shape, chroma_lst.shape)
    print("Shapes for: Arousal, Valence")
    print(arousal_lst.shape, valence_lst.shape)
    return data_lst, rhythm_lst, note_density_lst, arousal_lst, valence_lst, chroma_lst


class YamahaDataset(Dataset):
    '''
    Yamaha Piano e-competition dataset loader. No arousal/valence labels.
    '''
    def __init__(self, data, rhythm, note, chroma, mode="train"):
        super().__init__()
        inputs = data, rhythm, note, chroma
        indexed = []

        # train test split
        tlen, vlen = int(0.8 * len(data)), int(0.9 * len(data))

        for input in inputs:
            if mode == "train":
                indexed.append(input[:tlen])
            elif mode == "val":
                indexed.append(input[tlen:vlen])
            elif mode == "test":
                indexed.append(input[vlen:])

        self.data, self.rhythm, self.note, self.chroma = indexed
        self.r_density = [Counter(k)[1] / len(k) for k in self.rhythm]
        self.n_density = np.array([sum(k) / len(k) for k in self.note])
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        r = self.rhythm[idx]
        n = self.note[idx]
        c = self.chroma[idx]
        
        r_density = self.r_density[idx]
        n_density = self.n_density[idx]
        
        return x, r, n, c, r_density, n_density


class VGMIDIDataset(Dataset):
    '''
    VGMIDI dataset loader.
    '''
    def __init__(self, data, rhythm, note, chroma, arousal, valence, mode="train"):
        super().__init__()
        inputs = data, rhythm, note, chroma, arousal, valence
        indexed = []

        tlen, vlen = int(0.9 * len(data)), int(0.95 * len(data))

        for input in inputs:
            if mode == "train":
                indexed.append(input[:tlen])
            elif mode == "val":
                indexed.append(input[tlen:vlen])
            elif mode == "test":
                indexed.append(input[vlen:])

        self.data, self.rhythm, self.note, self.chroma, self.arousal, self.valence = indexed
        self.data = [torch.Tensor(np.insert(k, -1, 1)) for k in self.data]
        self.data = torch.nn.utils.rnn.pad_sequence(self.data, batch_first=True)

        # put this before applying torch.Tensor
        self.r_density = [Counter(list(k))[1] / len(k) for k in self.rhythm]
        self.n_density = np.array([sum(k) / len(k) for k in self.note])

        self.rhythm = [torch.Tensor(k) for k in self.rhythm]
        self.note = [torch.Tensor(k) for k in self.note]
        
        self.rhythm = torch.nn.utils.rnn.pad_sequence(self.rhythm, batch_first=True)
        self.note = torch.nn.utils.rnn.pad_sequence(self.note, batch_first=True)

        self.arousal[self.arousal >= 0] = 1
        self.arousal[self.arousal < 0] = 0
           
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        r = self.rhythm[idx]
        n = self.note[idx]
        c = self.chroma[idx]
        a = self.arousal[idx]
        v  =self.valence[idx]
        
        r_density = self.r_density[idx]
        n_density = self.n_density[idx]
        
        return x, r, n, c, a, v, r_density, n_density
