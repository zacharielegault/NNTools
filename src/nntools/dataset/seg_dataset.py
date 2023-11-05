import glob
import logging

import cv2
import numpy as np

from nntools import MISSING_DATA_FLAG, NN_FILL_DOWNSAMPLE, NN_FILL_UPSAMPLE
from nntools.dataset.image_tools import resize
from nntools.utils.io import path_leaf, read_image
from nntools.utils.misc import to_iterable

from .abstract_image_dataset import AbstractImageDataset, supportedExtensions


def extract_filename_without_extension(filename):
    return filename.split(".")[0]


class SegmentationDataset(AbstractImageDataset):
    def __init__(
        self,
        img_url,
        mask_url=None,
        shape=None,
        keep_size_ratio=False,
        recursive_loading=True,
        n_classes=None,
        extract_image_id_function=None,
        use_cache=False,
        auto_pad=True,
        filling_strategy=NN_FILL_UPSAMPLE,
        flag=cv2.IMREAD_COLOR,
        binarize_mask=False,
    ):
        if mask_url is None or mask_url == "":
            self.path_masks = None
        elif not isinstance(mask_url, dict):
            self.path_masks = {"mask": to_iterable(mask_url)}
        else:
            self.path_masks = {k: to_iterable(path) for k, path in mask_url.items()}
        if extract_image_id_function is None:
            extract_image_id_function = extract_filename_without_extension
            
        self.use_masks = self.path_masks is not None
        self.n_classes = n_classes
        self.filling_strategy = filling_strategy
        self.binarize_mask = binarize_mask

        super().__init__(
            img_url,
            shape,
            keep_size_ratio,
            recursive_loading,
            extract_image_id_function,
            auto_pad=auto_pad,
            use_cache=use_cache,
            flag=flag,
        )

    def get_class_count(self, save=False, load=False):
        from .utils import get_segmentation_class_count

        return get_segmentation_class_count(self, save=save, load=load)

    def list_files(self, recursive):
        for extension in supportedExtensions:
            prefix = "**/*." if recursive else "*."
            for path in self.path_img:
                self.img_filepath["image"].extend(glob.glob(path + prefix + extension, recursive=recursive))
            if self.use_masks:
                for mask_label, paths in self.path_masks.items():
                    for path in paths:
                        if mask_label not in self.gts:
                            self.gts[mask_label] = []
                        self.gts[mask_label].extend(glob.glob(path + prefix + extension, recursive=recursive))

        if self.use_masks:
            gts_ids = {}
            for mask_key in self.path_masks.keys():
                self.gts[mask_key] = np.asarray(self.gts[mask_key])
                gts_ids[mask_key] = [self.extract_image_id_function(path_leaf(path)) for path in self.gts[mask_key]]
                argsort_ids = np.argsort(gts_ids[mask_key])
                gts_ids[mask_key] = np.asarray(gts_ids[mask_key])[argsort_ids]
                self.gts[mask_key] = self.gts[mask_key][argsort_ids]

        self.img_filepath["image"] = np.asarray(self.img_filepath["image"])
        img_ids = np.asarray([self.extract_image_id_function(path_leaf(path)) for path in self.img_filepath["image"]])
        argsort_ids = np.argsort(img_ids)
        img_ids = img_ids[argsort_ids]
        self.img_filepath["image"] = self.img_filepath["image"][argsort_ids]

        if self.use_masks:
            list_lengths = [len(mask_ids) for mask_ids in gts_ids.values()] + [len(img_ids)]
            all_equal = all(elem == list_lengths[0] for elem in list_lengths)

            if not all_equal:
                logging.warning(
                    "Mismatch between the size of the different input folders (longer %i, smaller %i)"
                    % (max(list_lengths), min(list_lengths))
                )
                logging.debug(f"List lengths: {list(zip(list(gts_ids.keys())+['image'], list_lengths))}")

            list_common_file = set(img_ids)
            for mask_ids in gts_ids.values():
                list_common_file = list_common_file & set(mask_ids)
            intersection_ids = np.asarray(list(list_common_file))
            logging.debug(f"Number of files in intersection dataset: {len(intersection_ids)}")
            if len(intersection_ids) == 0:
                logging.warning("No common files between the different folders")
                for k in self.gts.keys():
                    logging.debug(f"List of files in {k}: {gts_ids[k]}")
                logging.debug(f"List of files in image: {img_ids}")

            if self.filling_strategy == NN_FILL_DOWNSAMPLE or all_equal:
                # We only keep the intersection of the files
                if not all_equal:
                    logging.warning("Downsampling the dataset to size %i" % min(list_lengths))

                self.img_filepath["image"] = self.img_filepath["image"][np.isin(img_ids, intersection_ids)]

                for k in self.gts.keys():
                    self.gts[k] = self.gts[k][np.isin(gts_ids[k], intersection_ids)]

            elif self.filling_strategy == NN_FILL_UPSAMPLE and not all_equal:
                if len(img_ids) < max(list_lengths):
                    raise ValueError("Upsampling is not possible if the dataset is smaller than the biggest folder")

                logging.warning("Upsampling missing labels to fit the dataset's size (%i)" % len(img_ids))
                for k, values in self.gts.items():
                    temps_ids = np.isin(img_ids, gts_ids[k])
                    gts_k = np.zeros(len(img_ids), dtype=values.dtype)
                    gts_k[temps_ids] = values
                    gts_k[~temps_ids] = MISSING_DATA_FLAG
                    self.gts[k] = gts_k

    def load_image(self, item):
        inputs = super(SegmentationDataset, self).load_image(item)
        actual_shape = inputs["image"].shape
        if self.use_masks:
            for k, file_list in self.gts.items():
                filepath = file_list[item]
                if filepath == MISSING_DATA_FLAG:
                    mask = np.zeros(actual_shape[:-1], dtype=np.uint8)
                else:
                    mask = read_image(filepath, cv2.IMREAD_GRAYSCALE)
                mask = self.resize_and_pad(mask, interpolation=cv2.INTER_NEAREST_EXACT)
                if self.binarize_mask:
                    inputs[k] = mask > 0
                else:
                    inputs[k] = mask.astype(np.uint8)

        return inputs

    def get_mask(self, item):
        filepath = self.gts[item]
        mask = read_image(filepath, cv2.IMREAD_GRAYSCALE)
        if self.auto_resize:
            mask = resize(image=mask, shape=self.shape, keep_size_ratio=self.keep_size_ratio, flag=cv2.INTER_NEAREST)
        if self.composer:
            mask = self.composer(mask=mask)
        if self.return_indices:
            return mask, item
        else:
            return mask
