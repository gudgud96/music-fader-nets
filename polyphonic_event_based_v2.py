import pypianoroll
import numpy as np
import os
from tqdm import tqdm

# 0-127 note on, 128 start token, 129 end token, 130 shift, 131-258 note off
# 259 empty, 260 - 387 velocity token

OFFSET_DISPLACEMENT = 131        # note off
VELOCITY_DISPLACEMENT = 260


def convert_pr_to_pitch_lst(pr):
    output_idx = []
    velocity_idx = []
    
    for i in range(len(pr)):
        indices = np.argwhere(pr[i] > 0)
        if len(indices) > 0:
            if len(indices) == 1:
                indices = np.expand_dims(indices.squeeze(), axis=0)
            else:
                indices = indices.squeeze()
        indices = list(indices)
        velocity = [pr[i][j] for j in indices]
        output_idx.append(indices)
        velocity_idx.append(velocity)
    
    return output_idx, velocity_idx


def pr_to_events(pitch_lst, velocity_lst):
    holding_pitches = sorted(pitch_lst[0])
    events = []
    vel_dict = {}

    # initialize
    for h in holding_pitches:
        idx = pitch_lst[0].index(h)
        events.append(note_on(h))
        events.append(vel(velocity_lst[0][idx]))
        vel_dict[h] = velocity_lst[0][idx]
    events.append(shift())

    for i in range(1, len(pitch_lst)):
        cur = pitch_lst[i]

        # find note-offs
        note_offs = sorted([k for k in holding_pitches if k not in cur])
        for n in note_offs:
            events.append(note_off(n))
            holding_pitches.remove(n)
        
        # find notes that are played with different velocity
        to_note_on = []
        for j in range(len(cur)):
            pitch, velocity = cur[j], velocity_lst[i][j]
            if pitch in holding_pitches and velocity != vel_dict[pitch]:
                events.append(note_off(pitch))
                holding_pitches.remove(pitch)
                to_note_on.append(pitch)

        # find new note-ons
        note_ons = sorted([k for k in cur if k not in holding_pitches])
        note_ons = sorted(note_ons + to_note_on)
        for n in note_ons:
            idx = pitch_lst[i].index(n)
            events.append(note_on(n))
            events.append(vel(velocity_lst[i][idx]))
            vel_dict[n] = velocity_lst[i][idx]
            holding_pitches.append(n)
        
        holding_pitches = sorted(holding_pitches)
        events.append(shift())
    
    # note-off remaining pitches
    for h in holding_pitches:
        events.append(note_off(h))
        holding_pitches.remove(h)
    
    return events


def events_to_pitch_lst(events):
    pitch_lst = []
    velocity_lst = []
    cur = []
    vel_dict = {}
    prev_onset = 0
    
    for e in events:
        if e == 130:
            cur_set = sorted(list(set(cur.copy())))
            if 0 in cur_set:
                cur_set.remove(0)
            vel_set = []
            for c in cur_set:
                if c in vel_dict:
                    vel_set.append(vel_dict[c])
                else:
                    vel_set.append(100)         # default velocity
            pitch_lst.append(cur_set)
            velocity_lst.append(vel_set)
        
        elif e == 128 or e == 129 or e == 259:
            continue
        
        else:
            if e - OFFSET_DISPLACEMENT < 0:     # it is an onset
                cur.append(e)
                prev_onset = e
            elif e - OFFSET_DISPLACEMENT in cur:      # it is an offset
                cur.remove(e - OFFSET_DISPLACEMENT)
            elif e - VELOCITY_DISPLACEMENT > 0:       # it is a velocity token
                if prev_onset in cur:  
                    vel_dict[prev_onset] = e - VELOCITY_DISPLACEMENT
                else:
                    print("Invalid token: {}".format(e))
                    pass
            else:
                print("Invalid token: {}".format(e))
                pass

    return pitch_lst, velocity_lst


def pitch_lst_to_pr(pitch_lst, velocity_lst):
    pr = []
    for i in range(len(pitch_lst)):
        p = pitch_lst[i]
        col = np.zeros(128,)
        for pitch in p:
            idx = p.index(pitch)
            col[pitch] = velocity_lst[i][idx]
        pr.append(col)
    pr = np.array(pr)
    return pr


def pitch_lst_to_rhythm(output_idx):    
    rhythm_lst = []
    if len(output_idx[0]) > 0:
        rhythm_lst.append(1)
    else:
        rhythm_lst.append(0)
    prev = output_idx[0]

    for i in range(1, len(output_idx)):
        if len(output_idx[i]) == 0:
            rhythm_lst.append(0)   # rest
        elif output_idx[i] == prev or all(elem in prev for elem in output_idx[i]):
            rhythm_lst.append(2)   # hold
        else:
            rhythm_lst.append(1)
        prev = output_idx[i]

    ret = rhythm_lst
    return ret


def encode_midi(fname, beat=24, is_pr=False):
    if not is_pr:
        track = pypianoroll.parse(fname, beat_resolution=beat)
        pr = track.get_merged_pianoroll()[:beat*8]
    else:
        pr = fname
    pitch_lst, velocity_lst = convert_pr_to_pitch_lst(pr)
    rhythm = pitch_lst_to_rhythm(pitch_lst)
    events = pr_to_events(pitch_lst, velocity_lst)
    return events, pitch_lst, velocity_lst, pr, rhythm


def decode_events(events):
    pitch_lst, velocity_lst = events_to_pitch_lst(events)
    pr = pitch_lst_to_pr(pitch_lst, velocity_lst)
    return pr, pitch_lst, velocity_lst
    

def shift():
    return 130


def note_on(pitch):
    return pitch


def note_off(pitch):
    return pitch + OFFSET_DISPLACEMENT      # add a displacement value


def vel(velocity):
    return int(velocity) + VELOCITY_DISPLACEMENT    # add a displacement value


def parse_pretty_midi(
        pm,
        mode="max",
        algorithm="normal",
        binarized=False,
        skip_empty_tracks=True,
        collect_onsets_only=False,
        threshold=0,
        first_beat_time=None,
        beat_resolution=4
    ):
    """
    Parse a :class:`pretty_midi.PrettyMIDI` object. The data type of the
    resulting pianorolls is automatically determined (int if 'mode' is
    'sum', np.uint8 if `mode` is 'max' and `binarized` is False, bool if
    `mode` is 'max' and `binarized` is True).

    Parameters
    ----------
    pm : `pretty_midi.PrettyMIDI` object
        A :class:`pretty_midi.PrettyMIDI` object to be parsed.
    mode : {'max', 'sum'}
        A string that indicates the merging strategy to apply to duplicate
        notes. Default to 'max'.
    algorithm : {'normal', 'strict', 'custom'}
        A string that indicates the method used to get the location of the
        first beat. Notes before it will be dropped unless an incomplete
        beat before it is found (see Notes for more information). Defaults
        to 'normal'.

        - The 'normal' algorithm estimates the location of the first beat by
            :meth:`pretty_midi.PrettyMIDI.estimate_beat_start`.
        - The 'strict' algorithm sets the first beat at the event time of
            the first time signature change. Raise a ValueError if no time
            signature change event is found.
        - The 'custom' algorithm takes argument `first_beat_time` as the
            location of the first beat.

    binarized : bool
        True to binarize the parsed pianorolls before merging duplicate
        notes. False to use the original parsed pianorolls. Defaults to
        False.
    skip_empty_tracks : bool
        True to remove tracks with empty pianorolls and compress the pitch
        range of the parsed pianorolls. False to retain the empty tracks
        and use the original parsed pianorolls. Deafault to True.
    collect_onsets_only : bool
        True to collect only the onset of the notes (i.e. note on events) in
        all tracks, where the note off and duration information are dropped.
        False to parse regular pianorolls.
    threshold : int or float
        A threshold used to binarize the parsed pianorolls. Only effective
        when `binarized` is True. Defaults to zero.
    first_beat_time : float
        The location (in sec) of the first beat. Required and only effective
        when using 'custom' algorithm.

    Notes
    -----
    If an incomplete beat before the first beat is found, an additional beat
    will be added before the (estimated) beat starting time. However, notes
    before the (estimated) beat starting time for more than one beat are
    dropped.

    """
    if mode not in ("max", "sum"):
        raise ValueError("`mode` must be one of {'max', 'sum'}.")
    if algorithm not in ("strict", "normal", "custom"):
        raise ValueError(
            "`algorithm` must be one of {'normal', 'strict', 'custom'}."
        )
    if algorithm == "custom":
        if not isinstance(first_beat_time, (int, float)):
            raise TypeError(
                "`first_beat_time` must be int or float when "
                "using 'custom' algorithm."
            )
        if first_beat_time < 0.0:
            raise ValueError(
                "`first_beat_time` must be a positive number "
                "when using 'custom' algorithm."
            )

    # Set first_beat_time for 'normal' and 'strict' modes
    if algorithm == "normal":
        if pm.time_signature_changes:
            pm.time_signature_changes.sort(key=lambda x: x.time)
            first_beat_time = pm.time_signature_changes[0].time
        else:
            first_beat_time = pm.estimate_beat_start()
    elif algorithm == "strict":
        if not pm.time_signature_changes:
            raise ValueError(
                "No time signature change event found. Unable to set beat start "
                "time using 'strict' algorithm."
            )
        pm.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = pm.time_signature_changes[0].time

    # get tempo change event times and contents
    tc_times, tempi = pm.get_tempo_changes()
    arg_sorted = np.argsort(tc_times)
    tc_times = tc_times[arg_sorted]
    tempi = tempi[arg_sorted]

    beat_times = pm.get_beats(first_beat_time)
    # NOTE: Below might break without len() as beat_times does not seems to always
    # be a list
    if not len(beat_times):  # pylint: disable=C1801
        raise ValueError("Cannot get beat timings to quantize pianoroll.")
    beat_times.sort()

    n_beats = len(beat_times)
    n_time_steps = beat_resolution * n_beats

    # Parse downbeat array
    if not pm.time_signature_changes:
        downbeat = None
    else:
        downbeat = np.zeros((n_time_steps,), bool)
        downbeat[0] = True
        start = 0
        end = start
        for idx, tsc in enumerate(pm.time_signature_changes[:-1]):
            end += np.searchsorted(
                beat_times[end:], pm.time_signature_changes[idx + 1].time
            )
            start_idx = start * beat_resolution
            end_idx = end * beat_resolution
            stride = tsc.numerator * beat_resolution
            downbeat[start_idx:end_idx:stride] = True
            start = end

    # Build tempo array
    one_more_beat = 2 * beat_times[-1] - beat_times[-2]
    beat_times_one_more = np.append(beat_times, one_more_beat)
    bpm = 60.0 / np.diff(beat_times_one_more)
    tempo = np.tile(bpm, (1, 24)).reshape(-1,)

    # Parse pianoroll
    tracks = []
    for instrument in pm.instruments:
        if binarized:
            pianoroll = np.zeros((n_time_steps, 128), bool)
        elif mode == "max":
            pianoroll = np.zeros((n_time_steps, 128), np.uint8)
        else:
            pianoroll = np.zeros((n_time_steps, 128), int)

        pitches = np.array(
            [note.pitch for note in instrument.notes if note.end > first_beat_time]
        )
        note_on_times = np.array(
            [note.start for note in instrument.notes if note.end > first_beat_time]
        )
        beat_indices = np.searchsorted(beat_times, note_on_times) - 1
        remained = note_on_times - beat_times[beat_indices]
        ratios = remained / (
            beat_times_one_more[beat_indices + 1] - beat_times[beat_indices]
        )
        rounded = np.round((beat_indices + ratios) * beat_resolution)
        note_ons = rounded.astype(int)

        if collect_onsets_only:
            pianoroll[note_ons, pitches] = True
        elif instrument.is_drum:
            if binarized:
                pianoroll[note_ons, pitches] = True
            else:
                velocities = [
                    note.velocity
                    for note in instrument.notes
                    if note.end > first_beat_time
                ]
                pianoroll[note_ons, pitches] = velocities
        else:
            note_off_times = np.array(
                [
                    note.end
                    for note in instrument.notes
                    if note.end > first_beat_time
                ]
            )
            beat_indices = np.searchsorted(beat_times, note_off_times) - 1
            remained = note_off_times - beat_times[beat_indices]
            ratios = remained / (
                beat_times_one_more[beat_indices + 1] - beat_times[beat_indices]
            )
            note_offs = ((beat_indices + ratios) * beat_resolution).astype(int)

            for idx, start in enumerate(note_ons):
                end = note_offs[idx]
                velocity = instrument.notes[idx].velocity

                if velocity < 1:
                    continue
                if binarized and velocity <= threshold:
                    continue

                if 0 < start < n_time_steps:
                    if pianoroll[start - 1, pitches[idx]]:
                        pianoroll[start - 1, pitches[idx]] = 0
                if end < n_time_steps - 1:
                    if pianoroll[end, pitches[idx]]:
                        end -= 1

                if binarized:
                    if mode == "sum":
                        pianoroll[start:end, pitches[idx]] += 1
                    elif mode == "max":
                        pianoroll[start:end, pitches[idx]] = True
                elif mode == "sum":
                    pianoroll[start:end, pitches[idx]] += velocity
                elif mode == "max":
                    maximum = np.maximum(
                        pianoroll[start:end, pitches[idx]], velocity
                    )
                    pianoroll[start:end, pitches[idx]] = maximum

    return pianoroll


def main():
    # labelled_midi = ["../../labelled/pieces/midi/" + k for k in os.listdir("../../labelled/pieces/midi/")]
    labelled_midi = ["/data/classic-piano/" + k for k in os.listdir("/data/classic-piano/")]
    for i in tqdm(range(len(labelled_midi))):
        fname = labelled_midi[i]
        events, pitch_lst, velocity_lst, pr_ori, rhythm = encode_midi(fname)
        pr, pitch_lst_2, velocity_lst_2 = decode_events(events)
        assert ((pr_ori == pr).all()) == True


if __name__ == "__main__":
    main()
