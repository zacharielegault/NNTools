import copy
import os

import cv2
import numpy as np
from torch import randperm, default_generator
from torch._utils import _accumulate

from nntools.tracker.warnings import Tracker


def get_classification_class_count(dataset, save=False, load=False):
    gts = dataset.gts
    unique, count = np.unique(gts)
    return count

def get_segmentation_class_count(dataset, save=True, load=True):
    shape = dataset.shape
    path = dataset.path_masks
    filepath = os.path.join(path, 'classe_count.npy')

    if os.path.isfile(filepath) and load:
        return np.load(filepath)
    list_masks = dataset.gts

    classes_counts = np.zeros(1024, dtype=int)  # Arbitrary large number (nb classes unknown at this point)

    for f in list_masks:
        mask = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, dsize=shape, interpolation=cv2.INTER_NEAREST)
        u, counts = np.unique(mask, return_counts=True)
        classes_counts[u] += counts

    classes_counts = classes_counts[:np.max(np.nonzero(classes_counts)) + 1]
    if save:
        np.save(filepath, classes_counts)
        Tracker.warn('Weights stored in ' + filepath)
    return classes_counts


def class_weighting(class_count, mode='balanced', ignore_index=-100, eps=1, log_smoothing=1.01):
    assert mode in ['balanced', 'log_prob']
    if mode == 'balanced':
        n_samples = sum([c for i, c in enumerate(class_count) if i != ignore_index])
        n_classes = len(np.nonzero(class_count))
        class_weights = n_samples / (n_classes * class_count + eps)

    elif mode == 'log_prob':
        p_class = class_count / class_count.sum()
        class_weights = (1 / np.log(log_smoothing + p_class)).astype(np.float32)
    if ignore_index >= 0:
        class_weights[ignore_index] = 0

    return class_weights.astype(np.float32)


def random_split(dataset, lengths, generator=default_generator):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    datasets = []
    for offset, length in zip(_accumulate(lengths), lengths):
        d = copy.deepcopy(dataset)
        indx = indices[offset - length: offset]
        d.subset(indx)
        datasets.append(d)
    return tuple(datasets)
