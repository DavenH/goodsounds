import os
import time

import torch
import torchaudio
from torch.utils.data import Dataset
import sqlite3
from alive_progress import alive_bar
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample

def to_midi(note, octave):
    # Define the mapping of notes to MIDI note numbers
    note_to_number = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}

    # Check if the input note is valid
    if note not in note_to_number:
        raise ValueError(f"Invalid note: {note}")

    # Calculate the MIDI note number based on the note and octave
    midi_note = note_to_number[note] + (octave + 1) * 12
    return midi_note

class GoodSoundsDataset(Dataset):
    def __init__(self, root_path: str, trunc_len: int, width: int, height: int):
        self.conn = sqlite3.connect(os.path.join(root_path, 'database.sqlite'))
        self.root_path = root_path
        self.cursor = self.conn.cursor()
        self.trunc_len = trunc_len

        self.cursor.execute("SELECT DISTINCT instrument from sounds WHERE instrument != ''")
        self.categories = [inst[0] for inst in self.cursor.fetchall()]
        self.map_to_index = dict([(category, i) for (i, category) in enumerate(self.categories)])

        self.data_cache = list()
        query = """SELECT s.id, s.instrument, s.semitone, t.filename
FROM sounds s
JOIN takes t ON s.id = t.sound_id
"""
        self.cursor.execute(query)
        self.ids_and_paths = self.cursor.fetchall()
        print(f"Dataset size: {len(self.ids_and_paths)}")

        self.resamplers = dict()
        samp_rate = 16000
        for orig_freq in [44100, 48000]:
            self.resamplers[orig_freq] = Resample(orig_freq=orig_freq, new_freq=samp_rate)

        hop_length = trunc_len // width

        self.mel_spectrogram_transform = MelSpectrogram(
            sample_rate=samp_rate,
            n_fft=height * 4,
            hop_length=hop_length,
            n_mels=height + 1 # plus 1 due to dc offset
        )
        self.amplitude_to_db_transform = AmplitudeToDB()

        # 16308 items in the dataset; we can probably load it all into memory
        # 128 * 256 * 4 = 2GB
        # perhaps even video memory - we'd need to load it all upfront then transfer to cuda device
        self.preload()

    def preload(self):
        print("Loading dataset into memory...")
        start_time = time.time_ns()

        with alive_bar(len(self.ids_and_paths), force_tty=True) as bar:
            for idx, (id, instrument, note, path) in enumerate(self.ids_and_paths):
                full_path = os.path.join(self.root_path, path)
                audio, sr = torchaudio.load(full_path)

                # Resample to 16kHz
                audio = self.resamplers[int(sr)](audio)

                # If the audio has more than one channel (is stereo), make it mono by selecting one channel
                if audio.shape[0] > 1:
                    audio = audio[0, :].unsqueeze(0)

                # trim or pad to ~2 seconds
                if audio.shape[1] < self.trunc_len:
                    audio = torch.nn.functional.pad(audio, (0, self.trunc_len - audio.shape[1]))
                else:
                    # Truncate
                    audio = audio[:, :self.trunc_len]

                mel_spec = self.mel_spectrogram_transform(audio)

                # skip the DC offset 'frequency' bin, it's all zero. Now we're width x height again
                # also skip one of the time bins, I don't know why but there's one more than expected there
                mel_spec = mel_spec[:,1:,1:]

                # transform to decibels, and scale it back to [0-1]
                log_mel_spec = (self.amplitude_to_db_transform(mel_spec) + 100) * 0.01

                self.data_cache.append(
                    {
                        'spectrogram': log_mel_spec,
                        'index': self.map_to_index[instrument],
                        'metadata': {
                            'path': path,
                            'instrument': instrument,

                            # we may want to predict this too
                            'note': note
                        }
                    }
                )

                time.sleep(0.001)
                bar()
                # if idx % 1000 == 0:
                #     print(f"Loaded {idx} of {len(self.ids_and_paths)} samples")
            end_time = time.time_ns()
        print(f"Done loading in {(end_time - start_time)/1e9:5.03f}s")


    def __len__(self):
        # Return the total number of sound samples
        return len(self.ids_and_paths)

    def __getitem__(self, idx):
        return self.data_cache[idx]
