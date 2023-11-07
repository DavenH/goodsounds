import os

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import sqlite3
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample


class GoodSoundsDataset(Dataset):
    def __init__(self, root_path: str, trunc_len: int, width: int, height: int):
        # Connect to the SQLite database
        self.conn = sqlite3.connect(os.path.join(root_path, 'database.sqlite'))
        self.root_path = root_path
        self.cursor = self.conn.cursor()
        self.trunc_len = trunc_len

        self.cursor.execute("SELECT COUNT(DISTINCT instrument) from sounds WHERE instrument != ''")
        self.n_categories = self.cursor.fetchone()

        query = """SELECT s.id, t.filename
FROM sounds s
JOIN takes t ON s.id = t.sound_id
"""
        self.cursor.execute(query)
        self.ids_and_paths = self.cursor.fetchall()

        self.resamplers = dict()
        samp_rate = 16000
        for orig_freq in [44100, 48000]:
            self.resamplers[orig_freq] = Resample(orig_freq=orig_freq, new_freq=samp_rate)


        hop_length = trunc_len // width

        self.mel_spectrogram_transform = MelSpectrogram(
            sample_rate=samp_rate,
            n_fft=1024,
            hop_length=hop_length,
            n_mels=height
        )
        self.amplitude_to_db_transform = AmplitudeToDB()

    def __len__(self):
        # Return the total number of sound samples
        return len(self.ids_and_paths)

    def __getitem__(self, idx):
        id, path = self.ids_and_paths[idx]

        # Fetch the sound metadata using the sound_id from your database
        sound_data = self.cursor.execute("SELECT instrument, note, octave FROM sounds WHERE id=?", (id, )).fetchone()

        full_path = os.path.join(self.root_path, path)
        audio, sr = torchaudio.load(full_path)

        # Resample to 16kHz
        audio = self.resamplers[int(sr)](audio)

        # If the audio has more than one channel (is stereo), make it mono by selecting one channel
        if audio.shape[0] > 1:
            audio = audio[0, :].unsqueeze(0)

        # trim or pad to ~4 seconds
        if audio.shape[1] < self.trunc_len:
            audio = torch.nn.functional.pad(audio, (0, self.trunc_len - audio.shape[1]))
        else:
            # Truncate
            audio = audio[:, :self.trunc_len]

        mel_spec = self.mel_spectrogram_transform(audio)
        log_mel_spec = self.amplitude_to_db_transform(mel_spec)

        # Return a dictionary of the data for this sound file
        return {
            'spectrogram': log_mel_spec,
            'audio': audio,
            'label': sound_data
        }

