

import copy

from torch import default_generator, randperm
from torch._utils import _accumulate


def random_split(dataset, lengths, generator=default_generator):
    if sum(lengths) == 1:
        lengths = [int(length * len(dataset)) for length in lengths[:-1]]
        lengths.append(len(dataset) - sum(lengths))  # To prevent rounding error

    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    datasets = []
    for offset, length in zip(_accumulate(lengths), lengths):
        d = copy.deepcopy(dataset)
        # We need to explicit call the attrs post init callback since deepcopy does not call it
        d.__attrs_post_init__()
        # We also need to explicitely copy the composer
        d.composer = copy.deepcopy(dataset.composer)
        indx = indices[offset - length : offset]
        d.subset(indx)

        datasets.append(d)
    return tuple(datasets)
