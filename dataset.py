import os
import sqlite3
import time

import pandas as pd
import torch
import torchaudio
from alive_progress import alive_bar
from torch.utils.data import Dataset
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB


class BaseAudioDataset(Dataset):
    def __init__(self, trunc_len, width, height):
        self.trunc_len = trunc_len
        samp_rate = 16000
        self.resamplers = {
            32000: Resample(orig_freq=32000, new_freq=samp_rate),
            44100: Resample(orig_freq=44100, new_freq=samp_rate),
            48000: Resample(orig_freq=48000, new_freq=samp_rate)
        }

        hop_length = trunc_len // width
        self.mel_spectrogram_transform = MelSpectrogram(
            sample_rate=samp_rate,
            n_fft=height * 4,
            hop_length=hop_length,
            n_mels=height + 1
        )

        self.amplitude_to_db_transform = AmplitudeToDB()
        self.data_cache = dict()

    def preprocess_audio(self, audio, sr):
        # Resample, convert to mono, trim or pad, and transform
        audio = self.resamplers[int(sr)](audio)
        if audio.shape[0] > 1:
            audio = audio[0, :].unsqueeze(0)
        if audio.shape[1] < self.trunc_len:
            audio = torch.nn.functional.pad(audio, (0, self.trunc_len - audio.shape[1]))
        else:
            audio = audio[:, :self.trunc_len]
        return audio

    def extract_features(self, audio):
        mel_spec = self.mel_spectrogram_transform(audio)
        log_mel_spec = (self.amplitude_to_db_transform(mel_spec) + 100) * 0.01
        return log_mel_spec[:,1:,1:]  # Adjusting dimensions as needed

    def preload(self):
        print("Loading dataset into memory...")
        start_time = time.time_ns()

        with alive_bar(len(self), force_tty=True) as bar:
            for i in range(len(self)):
                data = self.__getitem__(i)
                self.data_cache[i] = data
                time.sleep(0.001)
                bar()
        end_time = time.time_ns()
        print(f"Done loading in {(end_time - start_time)/1e9:5.03f}s")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class GoodSoundsDataset(BaseAudioDataset):
    def __init__(self, root_path: str, trunc_len: int, width: int, height: int):
        super().__init__(trunc_len, width, height)

        self.conn = sqlite3.connect(os.path.join(root_path, 'database.sqlite'))
        self.root_path = root_path
        self.cursor = self.conn.cursor()

        self.cursor.execute("SELECT DISTINCT instrument from sounds WHERE instrument != ''")
        self.categories = [inst[0] for inst in self.cursor.fetchall()]
        self.map_to_index = dict([(category, i) for (i, category) in enumerate(self.categories)])

        query = """SELECT s.id, s.instrument, s.semitone, t.filename
FROM sounds s
JOIN takes t ON s.id = t.sound_id
"""
        self.cursor.execute(query)
        self.ids_and_paths = self.cursor.fetchall()
        print(f"Dataset size: {len(self.ids_and_paths)}")

        # 16308 items in the dataset; we can probably load it all into memory
        # 128 * 256 * 4 = 2GB
        # perhaps even video memory - we'd need to load it all upfront then transfer to cuda device
        self.preload()

    def __len__(self):
        # Return the total number of sound samples
        return len(self.ids_and_paths)

    def __getitem__(self, idx):
        if idx in self.data_cache:
            return self.data_cache[idx]
        else:
            id, instrument, note, path = self.ids_and_paths[idx]
            full_path = os.path.join(self.root_path, path)
            audio, sr = torchaudio.load(full_path)
            audio = self.preprocess_audio(audio, sr)
            mel_spec = self.extract_features(audio)

            return {
                    'spectrogram': mel_spec,
                    'index': self.map_to_index[instrument],
                    'metadata': {
                        'path': path,
                        'label': instrument,

                        # we may want to predict this too
                        'note': note
                    }
                }


class FSDKaggle2019Dataset(BaseAudioDataset):
    def __init__(self, csv_file, root_dir, trunc_len, width, height):
        super().__init__(trunc_len, width, height)
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.label_map = {label: idx for idx, label in enumerate(self.annotations['labels'].unique())}

        # maybe this should split the dataset into a few chunks to fit into memory.
        self.preload()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if idx in self.data_cache:
            return self.data_cache[idx]
        else:
            audio_file = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
            waveform, sample_rate = torchaudio.load(audio_file)
            waveform = self.preprocess_audio(waveform, sample_rate)
            mel_spec = self.extract_features(waveform)

            label = self.annotations.iloc[idx, 1]
            label_index = self.label_map[label]

            return {
                'spectrogram': mel_spec,
                'index': label_index,
                'metadata': {
                    'path': audio_file,
                    'label': label
                }
            }
