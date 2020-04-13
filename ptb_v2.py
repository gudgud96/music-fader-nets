import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from tslearn.clustering import TimeSeriesKMeans
from tqdm import tqdm
import pretty_midi
# from utils import OrderedCounter
from collections import Counter
import sys, math
import pypianoroll
from polyphonic_event_based_v2 import *
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.preprocessing import StandardScaler
import music21

import magenta
print(magenta.__file__)
from magenta.models.score2perf.music_encoders import MidiPerformanceEncoder

from calculate_tension import calculate_diameter, extract_notes

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
    # midi_sample = mpe.decode(notes)
    # pm = pretty_midi.PrettyMIDI(midi_sample)
    pm = mpe.decode(notes, return_pm=True)
    return pm


def slice_midi(pm, beats, start_idx, end_idx):
    new_pm = pretty_midi.PrettyMIDI()
    new_inst = pretty_midi.Instrument(program=pm.instruments[0].program,
                                      is_drum=pm.instruments[0].is_drum,
                                      name=pm.instruments[0].name)
    start, end = beats[start_idx], beats[end_idx]
    for note in pm.instruments[0].notes:
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

    for ctrl in pm.instruments[0].control_changes:
        if ctrl.time >= start and ctrl.time < end:
            new_ctrl = pretty_midi.ControlChange(
                number=ctrl.number, value=ctrl.value, time=ctrl.time - start)
            new_inst.control_changes.append(new_ctrl)

    new_pm.instruments.append(new_inst)
    new_pm.write('tmp.mid')
    return new_pm


def get_harmony_vector(fname, is_one_hot=False):
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

        if not is_one_hot:
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


def get_dissonance_value(fname):
    pm, chord_names, chord_note, beats = extract_notes(fname, "./")
    _, result = calculate_diameter(fname, pm, beats, "./", window_size=1)
    return result


def get_music_attributes(pr, beat=24):
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


def get_classic_piano():

    labelled_midi = ["/data/classic-piano/" + k for k in os.listdir("/data/classic-piano/")]
    labelled_midi += ["/data/piano-e-competition/" + k for k in os.listdir("/data/piano-e-competition/")]
    # labelled_midi += ["/data/SUPRA/welte-red/midi-exp/" + k for k in os.listdir("/data/SUPRA/welte-red/midi-exp/")]

    print(len(labelled_midi))
    keylst = labelled_midi

    if not os.path.exists("values_v3/data.npy"):
        data_lst = []
        rhythm_lst = []
        note_density_lst = []
        tempo_change_lst = []
        velocity_lst = []
        chroma_lst = []

        key_signature_lst = []

        try:
            for i, name in tqdm(enumerate(keylst), total=len(keylst)):
                print(name, i, len(data_lst))
                beat_res = 4                # beat resolution
                num_of_beats = 4
                
                track = pypianoroll.parse(name, beat_resolution=beat_res).tracks
                
                if len(track) > 0:
                    try:
                        pm = pretty_midi.PrettyMIDI(name)
                        ks = pm.time_signature_changes[0]
                        beats = pm.get_beats()
                        tempo = pm.get_tempo_changes()
                        cur_idx, tempo_new = 0, []
                        for beat in beats:
                            while beat > tempo[0][cur_idx] and cur_idx < len(tempo[0]) - 1:
                                cur_idx += 1
                            tempo_new.append(tempo[1][cur_idx])
                        key_signature_lst.append("{}/{}".format(ks.numerator, ks.denominator))
                    except Exception as e:
                        print(e)
                        continue
                
                    pr = track[0].pianoroll

                    for j in range(0, len(pr), beat_res * num_of_beats):
                        start_idx = j
                        end_idx = j + beat_res * num_of_beats

                        if end_idx // beat_res < len(beats):
                            new_pr = pr[start_idx : end_idx]
                            new_pm = slice_midi(pm, beats, start_idx // beat_res, end_idx // beat_res)
                            new_pm.write("tmp.mid")
                            ms = np.argmax(new_pr, axis=-1)

                            if len(new_pm.instruments[0].notes) > 0 and \
                                len(np.unique(ms)) > 2 and np.count_nonzero(ms) >= 0.75 * len(ms):
                                
                                # get music attributes
                                _, rhythm, note_density, chroma, \
                                    velocity = get_music_attributes(new_pr, beat=beat_res)
                                tempo_lst = tempo_new[start_idx // beat_res: end_idx // beat_res]
                                tempo_lst = [int(k) for k in tempo_lst]

                                # get midi encoding sequence
                                events = magenta_encode_midi("tmp.mid")
                                is_eos = True
                                if is_eos:
                                    events.append(1)

                                if rhythm[0] == 1 and len(events) <= 100:   # filter out segments that start with 0
                                    chroma = get_harmony_vector()
                                    if chroma is None:
                                        continue
                                    else:
                                        # append music attributes
                                        data_lst.append(torch.Tensor(events))
                                        rhythm_lst.append(rhythm)
                                        note_density_lst.append(note_density)
                                        chroma_lst.append(chroma)

                                        print("{}/{}".format(start_idx // (beat_res * num_of_beats), len(pr) // (beat_res * num_of_beats)), end="\r")
        
        except Exception as e:
            print(e)
            print("Current dataset: {}".format(len(data_lst)))

        data_lst = torch.nn.utils.rnn.pad_sequence(data_lst, batch_first=True).numpy().astype(int)
        rhythm_lst = np.array(rhythm_lst)
        note_density_lst = np.array(note_density_lst)
        chroma_lst = np.array(chroma_lst)

        np.random.seed(777)
        idx = np.arange(len(data_lst))
        np.random.shuffle(idx)
        data_lst, rhythm_lst, note_density_lst, chroma_lst = data_lst[idx], \
                                                            rhythm_lst[idx], \
                                                            note_density_lst[idx], \
                                                            chroma_lst[idx]

        print(data_lst.shape, rhythm_lst.shape, note_density_lst.shape, chroma_lst.shape)

        np.save("values_v3/data.npy", data_lst)
        np.save("values_v3/rhythm.npy", rhythm_lst)
        np.save("values_v3/note_density.npy", note_density_lst)
        np.save("values_v3/chroma.npy", chroma_lst)

        print("Dataset saved!")
    
    else:
        data_lst = np.load("values_v3/data.npy")
        rhythm_lst = np.load("values_v3/rhythm.npy")
        note_density_lst = np.load("values_v3/note_density.npy")
        chroma_lst = np.load("values_v3/chroma.npy")

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

        print("Note density", np.amax(note_density_lst), np.amin(note_density_lst))
        print(data_lst.shape, rhythm_lst.shape, note_density_lst.shape, chroma_lst.shape)

        # get density normalization mean and variance
        n_density = [sum(k) / len(k) for k in note_density_lst]
        print(np.mean(n_density), np.std(n_density))

    return data_lst, rhythm_lst, note_density_lst, chroma_lst


def get_classic_piano_old():
    labelled_midi = ["/data/classic-piano/" + k for k in os.listdir("/data/classic-piano/")]
    labelled_midi += ["/data/piano-e-competition/" + k for k in os.listdir("/data/piano-e-competition/")]
    
    print(len(labelled_midi))
    keylst = labelled_midi

    if not os.path.exists("values_v2/data.npy"):
        data_lst = []
        rhythm_lst = []
        note_density_lst = []
        tempo_change_lst = []
        velocity_lst = []
        chroma_lst = []

        key_signature_lst = []

        try:
            def get_estimated_tempo(audio):
                write("example.wav", 44100, audio)
                audio = MonoLoader(filename="example.wav")()
                rhythm_extractor = RhythmExtractor2013(method="multifeature")
                bpm, _, _, _, _ = rhythm_extractor(audio)
                return bpm

            for i, name in tqdm(enumerate(keylst), total=len(keylst)):
                print(name, i, len(data_lst))
                beat_res = 4                # beat resolution
                num_of_beats = 4
                
                track = pypianoroll.parse(name, beat_resolution=beat_res).tracks
                
                if len(track) > 0:
                    try:
                        pm = pretty_midi.PrettyMIDI(name)
                        ks = pm.time_signature_changes[0]
                        beats = pm.get_beats()
                        tempo = pm.get_tempo_changes()
                        cur_idx, tempo_new = 0, []
                        for beat in beats:
                            while beat > tempo[0][cur_idx] and cur_idx < len(tempo[0]) - 1:
                                cur_idx += 1
                            tempo_new.append(tempo[1][cur_idx])
                        key_signature_lst.append("{}/{}".format(ks.numerator, ks.denominator))
                    except Exception as e:
                        print(e)
                        continue
                
                    pr = track[0].pianoroll

                    for j in range(0, len(pr), beat_res * num_of_beats):
                        start_idx = j
                        end_idx = j + beat_res * num_of_beats

                        if end_idx // beat_res < len(beats):
                            new_pr = pr[start_idx : end_idx]
                            new_pm = slice_midi(pm, beats, start_idx // beat_res, end_idx // beat_res)
                            ms = np.argmax(new_pr, axis=-1)

                            if len(new_pm.instruments[0].notes) > 0 and \
                                len(np.unique(ms)) > 2 and np.count_nonzero(ms) >= 0.75 * len(ms):
                                
                                # get music attributes
                                _, rhythm, note_density, chroma, \
                                    velocity = get_music_attributes(new_pr, beat=beat_res)
                                tempo_lst = tempo_new[start_idx // beat_res: end_idx // beat_res]
                                tempo_lst = [int(k) for k in tempo_lst]

                                # get midi encoding sequence
                                events = magenta_encode_midi("tmp.mid")
                                is_eos = True
                                if is_eos:
                                    events.append(1)

                                if rhythm[0] == 1 and len(events) <= 100:   # filter out segments that start with 0
                                    
                                    # get tempo using estimation from audio
                                    # reason: often times midi meta message stores a fix tempo value, which
                                    # we can't infer the true tempo value.
                                    audio = new_pm.fluidsynth()
                                    bpm = int(get_estimated_tempo(audio))
                                    
                                    # append music attributes
                                    data_lst.append(torch.Tensor(events))
                                    rhythm_lst.append(rhythm)
                                    note_density_lst.append(note_density)
                                    tempo_change_lst.append(bpm)
                                    velocity_lst.append(velocity)
                                    chroma_lst.append(chroma)

                                    print("{}/{}".format(start_idx // (beat_res * num_of_beats), len(pr) // (beat_res * num_of_beats)), end="\r")

                                    # print(events)
                                    # print(rhythm)
                                    # print(note_density)
                                    # print(bpm)
                                    # print(velocity)
                                    # print(chroma)
        
        except Exception as e:
            print(e)
            print("Current dataset: {}".format(len(data_lst)))

        data_lst = torch.nn.utils.rnn.pad_sequence(data_lst, batch_first=True).numpy().astype(int)
        rhythm_lst = np.array(rhythm_lst)
        tempo_change_lst = np.array(tempo_change_lst)
        note_density_lst = np.array(note_density_lst)
        velocity_lst = np.array(velocity_lst)
        chroma_lst = np.array(chroma_lst)
        chroma_lst[chroma_lst > 0] = 1

        max_density, max_event, max_velocity, max_tempo = np.amax(note_density_lst), \
                                                          np.amax(data_lst), \
                                                          np.amax(velocity_lst), \
                                                          np.amax(tempo_change_lst)
        
        print("Max", max_density, max_event, max_velocity, max_tempo)
        print(np.amax(tempo_change_lst), np.amin(tempo_change_lst))

        np.random.seed(777)
        idx = np.arange(len(data_lst))
        np.random.shuffle(idx)
        data_lst, rhythm_lst, tempo_change_lst, \
            note_density_lst, velocity_lst, chroma_lst = data_lst[idx], \
                                                         rhythm_lst[idx], \
                                                         tempo_change_lst[idx], \
                                                         note_density_lst[idx], \
                                                         velocity_lst[idx], \
                                                         chroma_lst[idx]

        print(data_lst.shape, rhythm_lst.shape, tempo_change_lst.shape)
        print(note_density_lst.shape, velocity_lst.shape, chroma_lst.shape)

        np.save("values_v2/data.npy", data_lst)
        np.save("values_v2/rhythm.npy", rhythm_lst)
        np.save("values_v2/tempo.npy", tempo_change_lst)
        np.save("values_v2/note_density.npy", note_density_lst)
        np.save("values_v2/velocity.npy", velocity_lst)
        np.save("values_v2/chroma.npy", chroma_lst)

        print("Dataset saved!")
    
    else:
        data_lst = np.load("values_v2/data.npy")
        rhythm_lst = np.load("values_v2/rhythm.npy")
        tempo_change_lst = np.load("values_v2/tempo.npy")
        note_density_lst = np.load("values_v2/note_density.npy")
        velocity_lst = np.load("values_v2/velocity.npy")
        chroma_lst = np.load("values_v2/chroma.npy")

        print(data_lst.shape, rhythm_lst.shape, tempo_change_lst.shape)
        print(note_density_lst.shape, velocity_lst.shape, chroma_lst.shape)
        
        print("Note", np.amax(note_density_lst), np.amin(note_density_lst))
        print("Tempo", np.amax(tempo_change_lst), np.amin(tempo_change_lst))
        print("Velocity", np.amax(velocity_lst), np.amin(velocity_lst))

        print(data_lst[0])

        length = 0
        for d in data_lst:
            length += len(np.trim_zeros(d))
        print("Avg length: ", length / len(data_lst))

        tempos = list(tempo_change_lst.reshape(-1))
        print(Counter(tempos))

    return data_lst, rhythm_lst, tempo_change_lst, note_density_lst, velocity_lst, chroma_lst


def get_classic_piano_v4():

    labelled_midi = ["/data/classic-piano/" + k for k in os.listdir("/data/classic-piano/")]
    labelled_midi += ["/data/piano-e-competition/" + k for k in os.listdir("/data/piano-e-competition/")]
    labelled_midi += ["/data/SUPRA/welte-red/midi-exp/" + k for k in os.listdir("/data/SUPRA/welte-red/midi-exp/")]

    print(len(labelled_midi))
    keylst = labelled_midi

    if not os.path.exists("values_v4/data.npy"):
        data_lst = []
        rhythm_lst = []
        note_density_lst = []
        tempo_change_lst = []
        velocity_lst = []
        chroma_lst = []
        dissonance_lst = []

        key_signature_lst = []

        for i, name in tqdm(enumerate(keylst), total=len(keylst)):
            try:
                # if i == 5: break
                print(name, i, len(data_lst))
                beat_res = 4                # beat resolution
                num_of_beats = 4
                
                track = pypianoroll.parse(name, beat_resolution=beat_res).tracks
                
                if len(track) > 0:
                    try:
                        pm = pretty_midi.PrettyMIDI(name)
                        ks = pm.time_signature_changes[0]
                        beats = pm.get_beats()
                        tempo = pm.get_tempo_changes()
                        cur_idx, tempo_new = 0, []
                        for beat in beats:
                            while beat > tempo[0][cur_idx] and cur_idx < len(tempo[0]) - 1:
                                cur_idx += 1
                            tempo_new.append(tempo[1][cur_idx])
                        key_signature_lst.append("{}/{}".format(ks.numerator, ks.denominator))
                    except Exception as e:
                        print(e)
                        continue
                
                    pr = track[0].pianoroll

                    for j in range(0, len(pr), beat_res * num_of_beats):
                        start_idx = j
                        end_idx = j + beat_res * num_of_beats

                        if end_idx // beat_res < len(beats):
                            new_pr = pr[start_idx : end_idx]
                            new_pm = slice_midi(pm, beats, start_idx // beat_res, end_idx // beat_res)
                            new_pm.write("tmp.mid")
                            ms = np.argmax(new_pr, axis=-1)

                            if len(new_pm.instruments[0].notes) > 0 and \
                                len(np.unique(ms)) > 2 and np.count_nonzero(ms) >= 0.75 * len(ms):
                                
                                # get music attributes
                                _, rhythm, note_density, chroma, \
                                    velocity = get_music_attributes(new_pr, beat=beat_res)
                                tempo_lst = tempo_new[start_idx // beat_res: end_idx // beat_res]
                                tempo_lst = [int(k) for k in tempo_lst]

                                # get midi encoding sequence
                                new_pm.write("tmp.mid")
                                events = magenta_encode_midi("tmp.mid")
                                is_eos = True
                                if is_eos:
                                    events.append(1)

                                if rhythm[0] == 1 and len(events) <= 100:   # filter out segments that start with 0
                                    new_pm.write("tmp.mid")
                                    chroma = get_harmony_vector("tmp.mid", is_one_hot=True)
                                    new_pm.write("tmp.mid")
                                    dissonance = get_dissonance_value("tmp.mid")
                                    if chroma is None:
                                        continue
                                    else:
                                        # append music attributes
                                        data_lst.append(torch.Tensor(events))
                                        rhythm_lst.append(rhythm)
                                        note_density_lst.append(note_density)
                                        chroma_lst.append(chroma)
                                        dissonance_lst.append(dissonance)

                                        print("{}/{}".format(start_idx // (beat_res * num_of_beats), len(pr) // (beat_res * num_of_beats)), end="\r")
        
            except Exception as e:
                print(e)
                print("Current dataset: {}".format(len(data_lst)))
                print("Index: {}".format(i))
                continue

        data_lst = torch.nn.utils.rnn.pad_sequence(data_lst, batch_first=True).numpy().astype(int)
        rhythm_lst = np.array(rhythm_lst)
        note_density_lst = np.array(note_density_lst)
        chroma_lst = np.array(chroma_lst)
        dissonance_lst = np.array(dissonance_lst)

        np.random.seed(777)
        idx = np.arange(len(data_lst))
        np.random.shuffle(idx)
        data_lst, rhythm_lst, note_density_lst, chroma_lst, dissonance_lst = data_lst[idx], \
                                                                            rhythm_lst[idx], \
                                                                            note_density_lst[idx], \
                                                                            chroma_lst[idx], \
                                                                            dissonance_lst[idx]

        print(data_lst.shape, rhythm_lst.shape, note_density_lst.shape, chroma_lst.shape)

        np.save("values_v4/data.npy", data_lst)
        np.save("values_v4/rhythm.npy", rhythm_lst)
        np.save("values_v4/note_density.npy", note_density_lst)
        np.save("values_v4/chroma.npy", chroma_lst)
        np.save("values_v4/dissonance.npy", dissonance_lst)

        print("Dataset saved!")
    
    else:
        data_lst = np.load("values_v4/data.npy")
        rhythm_lst = np.load("values_v4/rhythm.npy")
        note_density_lst = np.load("values_v4/note_density.npy")
        chroma_lst = np.load("values_v4/chroma.npy")
        dissonance_lst = np.load("values_v4/dissonance.npy")

        print(data_lst.shape, rhythm_lst.shape, note_density_lst.shape, chroma_lst.shape)

        c_r_lst, c_n_lst = [], []
        for r in rhythm_lst:
            r_density = Counter(r)[1] / len(r)
            if r_density < 0.3: c_r = 0
            elif r_density < 0.5: c_r = 1
            else: c_r = 2
            c_r_lst.append(c_r)

        for n in note_density_lst:
            n_density = sum(n) / len(n)
            if n_density <= 2: c_n = 0
            elif n_density <= 3.5: c_n = 1
            else: c_n = 2
            c_n_lst.append(c_n)
        
        print(Counter(c_r_lst))
        print(Counter(c_n_lst))
        print(np.amax(dissonance_lst), np.amin(dissonance_lst))

        length = 0
        for d in data_lst:
            length += len(np.trim_zeros(d))
        print("Avg length: ", length / len(data_lst))

    return data_lst, rhythm_lst, note_density_lst, chroma_lst


def get_classic_piano_v4_long():

    labelled_midi = ["/data/classic-piano/" + k for k in os.listdir("/data/classic-piano/")]
    labelled_midi += ["/data/piano-e-competition/" + k for k in os.listdir("/data/piano-e-competition/")]
    labelled_midi += ["/data/SUPRA/welte-red/midi-exp/" + k for k in os.listdir("/data/SUPRA/welte-red/midi-exp/")]

    print(len(labelled_midi))
    keylst = labelled_midi

    if not os.path.exists("values_v4_long/data.npy"):
        data_lst = []
        rhythm_lst = []
        note_density_lst = []
        tempo_change_lst = []
        velocity_lst = []
        chroma_lst = []
        dissonance_lst = []

        key_signature_lst = []

        for i, name in tqdm(enumerate(keylst), total=len(keylst)):
            try:
                # if i == 5: break
                print(name, i, len(data_lst))
                beat_res = 4                # beat resolution
                num_of_beats = 8
                
                track = pypianoroll.parse(name, beat_resolution=beat_res).tracks
                
                if len(track) > 0:
                    try:
                        pm = pretty_midi.PrettyMIDI(name)
                        ks = pm.time_signature_changes[0]
                        beats = pm.get_beats()
                        tempo = pm.get_tempo_changes()
                        cur_idx, tempo_new = 0, []
                        for beat in beats:
                            while beat > tempo[0][cur_idx] and cur_idx < len(tempo[0]) - 1:
                                cur_idx += 1
                            tempo_new.append(tempo[1][cur_idx])
                        key_signature_lst.append("{}/{}".format(ks.numerator, ks.denominator))
                    except Exception as e:
                        print(e)
                        continue
                
                    pr = track[0].pianoroll

                    for j in range(0, len(pr), beat_res * num_of_beats):
                        start_idx = j
                        end_idx = j + beat_res * num_of_beats

                        if end_idx // beat_res < len(beats):
                            new_pr = pr[start_idx : end_idx]
                            new_pm = slice_midi(pm, beats, start_idx // beat_res, end_idx // beat_res)
                            new_pm.write("tmp.mid")
                            ms = np.argmax(new_pr, axis=-1)

                            if len(new_pm.instruments[0].notes) > 0 and \
                                len(np.unique(ms)) > 2 and np.count_nonzero(ms) >= 0.75 * len(ms):
                                
                                # get music attributes
                                _, rhythm, note_density, chroma, \
                                    velocity = get_music_attributes(new_pr, beat=beat_res)
                                tempo_lst = tempo_new[start_idx // beat_res: end_idx // beat_res]
                                tempo_lst = [int(k) for k in tempo_lst]

                                # get midi encoding sequence
                                new_pm.write("tmp.mid")
                                events = magenta_encode_midi("tmp.mid")
                                is_eos = True
                                if is_eos:
                                    events.append(1)

                                if rhythm[0] == 1 and len(events) <= 100:   # filter out segments that start with 0
                                    new_pm.write("tmp.mid")
                                    chroma = get_harmony_vector("tmp.mid", is_one_hot=True)
                                    new_pm.write("tmp.mid")
                                    dissonance = get_dissonance_value("tmp.mid")
                                    if chroma is None:
                                        continue
                                    else:
                                        # append music attributes
                                        data_lst.append(torch.Tensor(events))
                                        rhythm_lst.append(rhythm)
                                        note_density_lst.append(note_density)
                                        chroma_lst.append(chroma)
                                        dissonance_lst.append(dissonance)

                                        print("{}/{}".format(start_idx // (beat_res * num_of_beats), len(pr) // (beat_res * num_of_beats)), end="\r")
        
            except Exception as e:
                print(e)
                print("Current dataset: {}".format(len(data_lst)))
                print("Index: {}".format(i))
                continue

        data_lst = torch.nn.utils.rnn.pad_sequence(data_lst, batch_first=True).numpy().astype(int)
        rhythm_lst = np.array(rhythm_lst)
        note_density_lst = np.array(note_density_lst)
        chroma_lst = np.array(chroma_lst)
        dissonance_lst = np.array(dissonance_lst)

        np.random.seed(777)
        idx = np.arange(len(data_lst))
        np.random.shuffle(idx)
        data_lst, rhythm_lst, note_density_lst, chroma_lst, dissonance_lst = data_lst[idx], \
                                                                            rhythm_lst[idx], \
                                                                            note_density_lst[idx], \
                                                                            chroma_lst[idx], \
                                                                            dissonance_lst[idx]

        print(data_lst.shape, rhythm_lst.shape, note_density_lst.shape, chroma_lst.shape)

        np.save("values_v4_long/data.npy", data_lst)
        np.save("values_v4_long/rhythm.npy", rhythm_lst)
        np.save("values_v4_long/note_density.npy", note_density_lst)
        np.save("values_v4_long/chroma.npy", chroma_lst)
        np.save("values_v4_long/dissonance.npy", dissonance_lst)

        print("Dataset saved!")
    
    else:
        data_lst = np.load("values_v4_long/data.npy")
        rhythm_lst = np.load("values_v4_long/rhythm.npy")
        note_density_lst = np.load("values_v4_long/note_density.npy")
        chroma_lst = np.load("values_v4_long/chroma.npy")
        dissonance_lst = np.load("values_v4_long/dissonance.npy")

        print(data_lst.shape, rhythm_lst.shape, note_density_lst.shape, chroma_lst.shape)

        c_r_lst, c_n_lst = [], []
        for r in rhythm_lst:
            r_density = Counter(r)[1] / len(r)
            if r_density < 0.3: c_r = 0
            elif r_density < 0.5: c_r = 1
            else: c_r = 2
            c_r_lst.append(c_r)

        for n in note_density_lst:
            n_density = sum(n) / len(n)
            if n_density <= 2: c_n = 0
            elif n_density <= 3.5: c_n = 1
            else: c_n = 2
            c_n_lst.append(c_n)
        
        print(Counter(c_r_lst))
        print(Counter(c_n_lst))
        print(np.amax(dissonance_lst), np.amin(dissonance_lst))

        length = 0
        for d in data_lst:
            length += len(np.trim_zeros(d))
        print("Avg length: ", length / len(data_lst))

    return data_lst, rhythm_lst, note_density_lst, chroma_lst


def get_vgmidi():
    data_lst = np.load("filtered_songs_disambiguate/song_tokens.npy", allow_pickle=True)
    rhythm_lst = np.load("filtered_songs_disambiguate/rhythm_lst.npy", allow_pickle=True)
    note_density_lst = np.load("filtered_songs_disambiguate/note_lst.npy", allow_pickle=True)
    valence_lst = np.load("filtered_songs_disambiguate/valence_lst.npy")
    arousal_lst = np.load("filtered_songs_disambiguate/arousal_lst.npy")

    if os.path.exists("filtered_songs_disambiguate/chroma_lst.npy"):
        chroma_lst = np.load("filtered_songs_disambiguate/chroma_lst.npy")
    else:
        chroma_lst = []
        for _, token in tqdm(enumerate(data_lst), total=len(data_lst)):
            pm = magenta_decode_midi(token)
            pm.write("vgmidi_tmp.mid")
            chroma = get_harmony_vector("vgmidi_tmp.mid", is_one_hot=True)
            chroma_lst.append(chroma)
        chroma_lst = np.array(chroma_lst)
        np.save("filtered_songs_disambiguate/chroma_lst.npy", chroma_lst)
    
    return data_lst, rhythm_lst, note_density_lst, arousal_lst, valence_lst, chroma_lst


class MusicAttrDataset(Dataset):
    def __init__(self, data, rhythm, tempo, note, velocity, chroma, mode="train"):
        super().__init__()
        inputs = data, rhythm, tempo, note, velocity, chroma
        indexed = []

        tlen, vlen = int(0.8 * len(data)), int(0.9 * len(data))

        for input in inputs:
            if mode == "train":
                indexed.append(input[:tlen])
            elif mode == "val":
                indexed.append(input[tlen:vlen])
            elif mode == "test":
                indexed.append(input[vlen:])

        self.data, self.rhythm, self.tempo, self.note, self.velocity, self.chroma = indexed
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        r = self.rhythm[idx]
        t = self.tempo[idx]
        n = self.note[idx]
        v = self.velocity[idx]
        c = self.chroma[idx]

        c_r, c_n, c_t, c_v = self.get_classes(r, n, t, v)
        
        return x, r, t, n, v, c, c_r, c_n, c_t, c_v
    
    def get_classes(self, r, n, t, v):
        r_density = Counter(r)[1] / len(r)
        if r_density < 0.25: c_r = 0
        elif r_density < 0.5: c_r = 1
        elif r_density < 0.75: c_r = 2
        else: c_r = 3

        n_density = sum(n) / len(n)
        if n_density <= 2: c_n = 0
        elif n_density <= 4: c_n = 1
        elif n_density <= 6: c_n = 2
        else: c_n = 3

        t_density = (t - MIN_TEMPO) / (MAX_TEMPO - MIN_TEMPO)
        if t_density < 0.25: c_t = 0
        elif t_density < 0.5: c_t = 1
        elif t_density < 0.75: c_t = 2
        else: c_t = 3

        v_density = sum(v) / len(v)
        v_density = (v_density - MIN_VELOCITY) / (MAX_VELOCITY - MIN_VELOCITY)
        if v_density < 0.25: c_v = 0
        elif v_density < 0.5: c_v = 1
        elif v_density < 0.75: c_v = 2
        else: c_v = 3

        return c_r, c_n, c_t, c_v


class MusicAttrDataset2(Dataset):
    def __init__(self, data, rhythm, note, chroma, mode="train"):
        super().__init__()
        inputs = data, rhythm, note, chroma
        indexed = []

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
        # scaler = StandardScaler()
        self.n_density = np.array([sum(k) / len(k) for k in self.note])
        # self.n_density = scaler.fit_transform(np.expand_dims(self.n_density, axis=-1)).squeeze()
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        r = self.rhythm[idx]
        n = self.note[idx]
        c = self.chroma[idx]
        
        r_density = self.r_density[idx]
        n_density = self.n_density[idx]

        c_r, c_n = self.get_classes(r, n)
        
        return x, r, n, c, c_r, c_n, r_density, n_density
    
    def get_classes(self, r, n):
        r_density = Counter(r)[1] / len(r)
        if r_density < 0.3: c_r = 0
        elif r_density < 0.5: c_r = 1
        else: c_r = 2

        n_density = sum(n) / len(n)
        if n_density <= 2: c_n = 0
        elif n_density <= 3.5: c_n = 1
        else: c_n = 2

        return c_r, c_n


class MusicAttrDataset3(Dataset):
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

        # self.labels = []
        # for i in range(len(self.arousal)):
        #     if self.arousal[i] >= 0 and self.valence[i] >= 0:
        #         self.labels.append(0)
        #     elif self.arousal[i] >= 0 and self.valence[i] < 0:
        #         self.labels.append(1)
        #     elif self.arousal[i] < 0 and self.valence[i] < 0:
        #         self.labels.append(2)
        #     else:
        #         self.labels.append(3)
        # print("Emotion labels:", Counter(self.labels))
        # self.labels = np.array(self.labels)
           
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        r = self.rhythm[idx]
        n = self.note[idx]
        c = self.chroma[idx]
        a = self.arousal[idx]
        # a = self.labels[idx]
        v  =self.valence[idx]
        
        r_density = self.r_density[idx]
        n_density = self.n_density[idx]
        
        return x, r, n, c, a, v, r_density, n_density



class MusicPolyphonic(Dataset):
    def __init__(self, data, rhythm, length):
        super().__init__()
        self.data, self.rhythm, self.length = data, rhythm, length
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        r = self.rhythm[idx]
        r_density = Counter(r)[1] / len(r)
        r_density_class = int(r_density * 8)
        if r_density_class > 3: r_density_class = 3
        d_len = self.length[idx]
        
        return d, torch.Tensor(r), r_density_class


class MusicPolyphonicNote(Dataset):
    def __init__(self, data, note_density_lst, length):
        super().__init__()
        self.data, self.note_density_lst, self.length = data, note_density_lst, length
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        n = self.note_density_lst[idx]
        n_density = sum(n) / len(n)
        if n_density < 2:
            n_density_class = 0
        elif n_density < 3:
            n_density_class = 1
        elif n_density < 5:
            n_density_class = 2
        else:
            n_density_class = 3
        
        return d, torch.Tensor(n), n_density_class

# get_classic_piano_v4_long()