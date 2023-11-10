import os

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import sqlite3
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
        # Connect to the SQLite database
        self.conn = sqlite3.connect(os.path.join(root_path, 'database.sqlite'))
        self.root_path = root_path
        self.cursor = self.conn.cursor()
        self.trunc_len = trunc_len

        self.cursor.execute("SELECT DISTINCT instrument from sounds WHERE instrument != ''")
        self.categories = [inst[0] for inst in self.cursor.fetchall()]
        self.map_to_index = dict([(self.categories[i], i) for i in range(len(self.categories))])

        query = """SELECT s.id, t.filename
FROM sounds s
JOIN takes t ON s.id = t.sound_id
"""
        self.cursor.execute(query)
        self.ids_and_paths = self.cursor.fetchall()
        print(len(self.ids_and_paths))
        self.resamplers = dict()
        samp_rate = 16000
        for orig_freq in [44100, 48000]:
            self.resamplers[orig_freq] = Resample(orig_freq=orig_freq, new_freq=samp_rate)


        hop_length = trunc_len // width

        self.mel_spectrogram_transform = MelSpectrogram(
            sample_rate=samp_rate,
            n_fft=height * 4,
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

        # trim or pad to ~2 seconds
        if audio.shape[1] < self.trunc_len:
            audio = torch.nn.functional.pad(audio, (0, self.trunc_len - audio.shape[1]))
        else:
            # Truncate
            audio = audio[:, :self.trunc_len]

        mel_spec = self.mel_spectrogram_transform(audio)

        # skip the DC offset, it's all zero. Now we're width x height again
        mel_spec = mel_spec[:,:,1:]

        log_mel_spec = (self.amplitude_to_db_transform(mel_spec) + 100) * 0.01

        # Return a dictionary of the data for this sound file
        return {
            'spectrogram': log_mel_spec,
            'audio': audio,
            'index': self.map_to_index[sound_data[0]],
            'metadata': {
                'instrument': sound_data[0],
                # we may want to predict this too
                'note': to_midi(sound_data[1], sound_data[2])
            }
        }

