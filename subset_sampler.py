import random
from typing import Optional, Sized

from torch.utils.data import Sampler
from dataset import BaseAudioDataset


class SubsetSampler(Sampler):
    def __init__(self, source: BaseAudioDataset, max_subset_samples, data_source: Optional[Sized]):
        super().__init__(data_source)
        self.source = source
        self.max_subset_samples = max_subset_samples
        size = len(self.source)
        self.subset_indices = self._make_indices(size)

    def refresh(self, state: dict):
        self.subset_indices = self._make_indices(len(self.source))
        print("Refreshing train dataset")

        yield from self.source.preload_indices(self.subset_indices, state)
        print("Done")

    def _make_indices(self, size: int):
        return random.sample(range(size), min(size, self.max_subset_samples))

    def __iter__(self):
        return iter(self.subset_indices)

    def __len__(self):
        return len(self.subset_indices)

    def load_state_dict(self, config: dict):
        self.subset_indices = config["indices"]

    def save_state_dict(self) -> dict:
        return dict(indices=self.subset_indices)
