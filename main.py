import os

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import sqlite3
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample


class GoodSoundsDataset(Dataset):
    def __init__(self, root_path):
        # Connect to the SQLite database
        self.conn = sqlite3.connect(os.path.join(root_path, 'database.sqlite'))
        self.root_path = root_path
        self.cursor = self.conn.cursor()

        query = """SELECT s.id, t.filename
FROM sounds s
JOIN takes t ON s.id = t.sound_id
"""
        self.cursor.execute(query)
        self.ids_and_paths = self.cursor.fetchall()

        self.resamplers = dict()
        for orig_freq in [44100, 48000]:
            self.resamplers[orig_freq] = Resample(orig_freq=orig_freq, new_freq=16000)

        self.mel_spectrogram_transform = MelSpectrogram(
            sample_rate=16000,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        self.amplitude_to_db_transform = AmplitudeToDB()

    def __len__(self):
        # Return the total number of sound samples
        return len(self.ids_and_paths)

    def __getitem__(self, idx):
        id, path = self.ids_and_paths[idx]

        # Fetch the sound metadata using the sound_id from your database
        sound_data = self.cursor.execute("SELECT instrument, note, octave FROM sounds WHERE id=?", (id, )).fetchone()

        # Here you would add the logic to load the audio file. For example:
        full_path = os.path.join(self.root_path, path)
        audio, sr = torchaudio.load(full_path)

        # Resample to 16kHz
        audio = self.resamplers[int(sr)](audio)

        # If the audio has more than one channel (is stereo), make it mono by selecting one channel
        if audio.shape[0] > 1:
            audio = audio[0, :].unsqueeze(0)

        audio_len = 65536
        # trim or pad to ~4 seconds
        if audio.shape[1] < audio_len:
            audio = torch.nn.functional.pad(audio, (0, audio_len - audio.shape[1]))
        else:
            # Truncate
            audio = audio[:, :audio_len]

        mel_spec = self.mel_spectrogram_transform(audio)
        log_mel_spec = self.amplitude_to_db_transform(mel_spec)

        # Return a dictionary of the data for this sound file
        return {
            'spectrogram': log_mel_spec,
            'audio': audio,
            'label': sound_data
        }


# Apply the dataset class
root_path = '/home/daven/data/audio/good-sounds'
good_sounds_dataset = GoodSoundsDataset(root_path=root_path)

# Create a DataLoader
dataloader = DataLoader(good_sounds_dataset, batch_size=4, shuffle=True)

# Iterate over the dataset
for i, data in enumerate(dataloader):
    print(data['label'])

    if i > 10:
        break
    # Here, you can send this data to your model
