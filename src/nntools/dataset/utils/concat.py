import bisect
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from nntools.dataset.abstract_image_dataset import AbstractImageDataset


def concat_datasets_if_needed(datasets):
    if isinstance(datasets, list):
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]
        return dataset
    else:
        return datasets


class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, *args, **kwargs):
        self.post_init = False
        self.datasets: list[AbstractImageDataset]
        super(ConcatDataset, self).__init__(*args, **kwargs)
        self.post_init = True

    def plot(self, idx, **kwargs):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        self.datasets[dataset_idx].plot(sample_idx, **kwargs)

    def get_class_count(self, load=True, save=True):
        class_count = None
        for d in self.datasets:
            if class_count is None:
                class_count = d.get_class_count(load=load, save=save)
            else:
                class_count += d.get_class_count(load=load, save=save)
        return class_count

    @property
    def composer(self):
        return [d.composer for d in self.datasets]

    def multiply_size(self, factor):
        for d in self.datasets:
            d.multiply_size(factor)

    def init_cache(self):
        for d in self.datasets:
            d.init_cache()

    def __setattr__(self, key, value):
        if key == "post_init":
            super(ConcatDataset, self).__setattr__(key, value)
        if hasattr(self, key) or not self.post_init:
            super(ConcatDataset, self).__setattr__(key, value)
        else:
            for d in self.datasets:
                d.__setattr__(key, value)