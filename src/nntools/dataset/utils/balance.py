
import logging
import os

import numpy as np
import tqdm


def get_segmentation_class_count(dataset, save=False, load=False):
    sample = dataset[0]
    if "mask" not in sample.keys():
        raise NotImplementedError

    path = dataset.path_img[0]
    filepath = os.path.join(path, "classes_count.npy")

    if os.path.isfile(filepath) and load:
        return np.load(filepath)
    classes_counts = np.zeros(1024, dtype=int)  # Arbitrary large number (nb classes unknown at this point)

    for sample in tqdm.tqdm(dataset):
        mask = sample["mask"].numpy()
        if mask.ndim == 3:  # Multilabel -> Multiclass
            arr_tmp = np.argmax(mask, axis=0) + 1
            arr_tmp[mask.max(axis=0) == 0] = 0
            mask = arr_tmp
        u, counts = np.unique(mask, return_counts=True)
        classes_counts[u] += counts

    classes_counts = classes_counts[: np.max(np.nonzero(classes_counts)) + 1]
    if save:
        np.save(filepath, classes_counts)
        logging.warn("Weights stored in " + filepath)
    return classes_counts


def class_weighting(class_count, mode="balanced", ignore_index=-100, eps=1, log_smoothing=1.01, center_mean=0):
    assert mode in ["balanced", "log_prob", "frequency"]
    n_samples = sum([c for i, c in enumerate(class_count) if i != ignore_index])

    if mode == "balanced":
        n_classes = len(np.nonzero(class_count))
        class_weights = n_samples / (n_classes * class_count + eps)
    elif mode == "frequency":
        class_weights = n_samples / class_count

    elif mode == "log_prob":
        p_class = class_count / n_samples
        class_weights = (1 / np.log(log_smoothing + p_class)).astype(np.float32)

    if center_mean:
        class_weights = class_weights - class_weights.mean() + center_mean
    if ignore_index >= 0:
        class_weights[ignore_index] = 0

    return class_weights.astype(np.float32)




