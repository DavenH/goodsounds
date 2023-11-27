import os
import sqlite3
import time
from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB

from events import load_progress_event, load_start_event, load_end_event


class BaseAudioDataset(Dataset):
    def __init__(self, sample_rate, trunc_len, width, height):
        self.trunc_len = trunc_len
        n_fft=height * 6

        self.resamplers = {
            32000: Resample(orig_freq=32000, new_freq=sample_rate),
            44100: Resample(orig_freq=44100, new_freq=sample_rate),
            48000: Resample(orig_freq=48000, new_freq=sample_rate)
        }

        hop_length = trunc_len // width
        self.mel_spectrogram_transform = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=height + 1
        )

        self.amplitude_to_db_transform = AmplitudeToDB()
        self.data_cache = dict()

        self.config = dict(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=height,
            hop_length=hop_length,
            trunc_len=trunc_len
        )

    def preprocess_audio(self, audio, sr, offset=0):
        # Resample, convert to mono, trim or pad, and transform
        if sr != self.config["sample_rate"]:
            audio = self.resamplers[int(sr)](audio)

        if audio.shape[0] > 1:
            audio = audio[0, :].unsqueeze(0)

        if audio.shape[1] < self.trunc_len:
            audio = torch.nn.functional.pad(audio, (0, self.trunc_len - audio.shape[1]))
        else:
            offset = min(offset, audio.shape[1] - self.trunc_len)
            audio = audio[:, offset:offset + self.trunc_len]
        return audio

    def extract_features(self, audio):
        mel_spec = self.mel_spectrogram_transform(audio)
        log_mel_spec = (self.amplitude_to_db_transform(mel_spec) + 100) * 0.01
        return log_mel_spec[:,1:,1:]  # Adjusting dimensions as needed

    def preload(self, state: dict):
        yield from self.preload_indices(range(len(self)), state)

    def preload_indices(self, indices: [int], state: dict):
        size = len(indices)
        print(f"Loading {size} elements into memory...")
        yield load_start_event(size)

        start_time = time.time_ns()
        self.data_cache = dict()
        # with alive_bar(size, force_tty=True) as bar:
        count = 0
        for i in indices:
            if state["paused"]:
                break
            data = self.__getitem__(i)
            self.data_cache[i] = data
            time.sleep(0.001)
            yield load_progress_event(count, size)
            count += 1
            # bar()

        end_time = time.time_ns()
        elapsed = (end_time - start_time)/1e9
        yield load_end_event(size, elapsed)

        print(f"Done loading in {elapsed:5.03f}s")

    def get_config(self):
        self.config["size"] = self.__len__()
        return self.config

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_loss_function(self):
        raise NotImplementedError

    def get_categories(self) -> List[str]:
        raise NotImplementedError


class GoodSounds(BaseAudioDataset):
    def __init__(self, root_path: str, sample_rate: int, trunc_len: int, width: int, height: int):
        super().__init__(sample_rate, trunc_len, width, height)

        self.conn = sqlite3.connect(os.path.join(root_path, 'database.sqlite'))
        self.root_path = root_path
        self.cursor = self.conn.cursor()

        self.cursor.execute("SELECT DISTINCT instrument from sounds WHERE instrument != ''")
        self.categories = [inst[0] for inst in self.cursor.fetchall()]
        self.label_map = dict([(category, i) for (i, category) in enumerate(self.categories)])

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
                'label': self.label_map[instrument],
                'metadata': {
                    'path': path,
                    'label': instrument,

                    # we may want to predict this too
                    'note': note
                }
            }

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def get_categories(self) -> List[str]:
        return self.categories


class FSDKaggle2019(BaseAudioDataset):
    def __init__(self, annotation_csv, wav_dir, sample_rate, trunc_len, data_per_wav, width, height):
        super().__init__(sample_rate, trunc_len, width, height)

        self.data_per_wav = data_per_wav
        self.wav_dir = wav_dir
        self.annotations = pd.read_csv(annotation_csv)

        # Get unique labels
        uniq_values = set()
        self.annotations['labels'].str.split(',').apply(uniq_values.update)
        self.categories = list(uniq_values)
        self.categories.sort()
        self.label_map = {label: idx for idx, label in enumerate(self.categories)}

    def __len__(self):
        # return 1000
        return len(self.annotations) * self.data_per_wav

    def __getitem__(self, idx):
        if idx in self.data_cache:
            return self.data_cache[idx]
        else:
            audio_file = os.path.join(self.wav_dir, self.annotations.iloc[idx // self.data_per_wav, 0])
            waveform, sample_rate = torchaudio.load(audio_file)
            offset = round((idx % self.data_per_wav) / self.data_per_wav * waveform.shape[1])
            waveform = self.preprocess_audio(waveform, sample_rate, offset)
            mel_spec = self.extract_features(waveform)

            label = self.annotations.iloc[idx // self.data_per_wav, 1]
            labels = label.split(',')
            label_indices = [self.label_map[label] for label in labels]
            binary_labels = torch.zeros(len(self.label_map))
            binary_labels[label_indices] = 1

            return {
                'spectrogram': mel_spec,
                'label': binary_labels,
                'metadata': {
                    'path': audio_file,
                    'label': label
                }
            }

    def get_loss_function(self):
        return nn.BCEWithLogitsLoss()

    def calc_accuracy(self, output, target):
        # Apply threshold to output to get binary predictions
        preds = output >= 0.0
        # Convert to byte tensors for comparison
        preds = preds.to(torch.uint8)
        target = target.to(torch.uint8)
        # Compare predictions with true labels
        correct = (preds == target).all(dim=1)  # Correct if all labels match
        # Calculate accuracy
        accuracy = correct.sum().float() / target.size(0)
        return accuracy

    def get_categories(self) -> List[str]:
        return self.categories