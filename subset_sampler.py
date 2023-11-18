import random

from torch.utils.data import Sampler

from dataset import BaseAudioDataset


class SubsetSampler(Sampler):
    def __init__(self, source: BaseAudioDataset, max_subset_samples):
        super().__init__()
        self.source = source
        self.max_subset_samples = max_subset_samples
        size = len(self.source)
        self.subset_indices = random.sample(range(size), min(size, self.max_subset_samples))
        self.refresh()

    def refresh(self):
        self.subset_indices = random.sample(range(len(self.source)), self.max_subset_samples)
        print("Refreshing train dataset")

        self.source.preload_indices(self.subset_indices)
        print("Done")

    def __iter__(self):
        return iter(self.subset_indices)

    def __len__(self):
        return len(self.subset_indices)