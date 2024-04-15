from typing import List

from nntools.dataset.abstract_image_dataset import AbstractImageDataset
from nntools.dataset.utils.concat import ConcatDataset


def check_dataleaks(*datasets: List[AbstractImageDataset], raise_exception=True):
    is_okay = True
    cols = {"files": [], "gts": []}
    unfold_datasets = []
    for d in datasets:
        if isinstance(d, ConcatDataset):
            unfold_datasets += d.datasets
        else:
            unfold_datasets.append(d)

    for d in unfold_datasets:
        file_cols, gts_cols = d.columns()
        cols["files"].append(list(file_cols))
        cols["gts"].append(list(gts_cols))

    cols["files"] = list(set.intersection(*map(set, cols["files"])))  # Find intersection of columns
    cols["gts"] = list(set.intersection(*map(set, cols["gts"])))

    for f_col in cols["files"]:
        filenames = []
        for d in unfold_datasets:
            filenames.append(d.filenames[f_col])
        join_file = list(set.intersection(*map(set, filenames)))
        if len(join_file) > 0:
            is_okay = False
            if raise_exception:
                raise ValueError("Found common images between datasets")

    for f_col in cols["gts"]:
        filenames = []
        for d in unfold_datasets:
            filenames.append(d.gt_filenames[f_col])
        join_gt = list(set.intersection(*map(set, filenames)))
        if len(join_gt) > 0:
            is_okay = False
            if raise_exception:
                raise ValueError("Found common groundtruth between datasets")

    if not is_okay:
        return is_okay, join_file, join_gt
    return is_okay