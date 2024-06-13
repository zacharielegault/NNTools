import logging
import multiprocessing as mp
from multiprocessing import shared_memory
import os

import numpy as np

from nntools.dataset.cache.abstract_cache import AbstractCache


def _get_rank() -> int:
    # Borrowed from pytorch_lightning
    # https://pytorch-lightning.readthedocs.io/en/1.7.7/_modules/pytorch_lightning/utilities/rank_zero.html#rank_zero_warn
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


class MemoryCache(AbstractCache):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.cache_with_shared_array = True
        self.cache_arrays = None
        self._is_first_process = False
        
    def init_cache(self):
        if self.is_initialized:
            return
        
        if not self.d.auto_resize and not self.d.auto_pad:
            logging.warning(
                "You are using a cache with auto_resize and auto_pad set to False.\
                    Make sure all your images are the same size"
            )
        
        arrays = self.d.read_from_disk(0)  # Taking the first element
        arrays = self.d.precompose_data(arrays)

        in_memory_arrays = dict()
        # Keep reference to all shm avoid the call from the garbage collector which pointer to buffer error
        if self.cache_with_shared_array:
            self.init_shared_items_tracking()
        else:
            self.init_non_shared_items_tracking()
            
        for key, arr in arrays.items():
            if not isinstance(arr, np.ndarray):
                in_memory_arrays[key] = np.ndarray(self.nb_samples, dtype=type(arr))
                continue

            data_shape = (self.nb_samples, *arr.shape)
            if self.cache_with_shared_array:
                try:
                    if _get_rank()==0:
                            shm = shared_memory.SharedMemory(
                                name=f"nntools_{key}_{self.id}", size=arr.nbytes * self.nb_samples, create=True
                            )
                            self._is_first_process = True
                            logging.info(f"Creating shared memory, {mp.current_process().name}")
                            logging.debug(f"nntools_{key}_{self.id}: size: {shm.buf.nbytes} ({data_shape})")
                            shared_array = np.frombuffer(buffer=shm.buf, dtype=arr.dtype).reshape(data_shape)
                            # The initialization with 0 is not needed.
                            # However, it's a good way to check if the shared memory is correctly initialized
                            # And it checks if there is enough space in dev/shm
                            shared_array[:] = 0
                    else:
                        shm = shared_memory.SharedMemory(name=f"nntools_{key}_{self.id}", create=False)
                        logging.info(f"Accessing existing shared memory {mp.current_process().name}")
                        shared_array = np.frombuffer(buffer=shm.buf, dtype=arr.dtype).reshape(data_shape)
                        
                except FileExistsError:
                    shm = shared_memory.SharedMemory(name=f"nntools_{key}_{self.id}", create=False)
                    logging.info(f"Accessing existing shared memory {mp.current_process().name}")
                    shared_array = np.frombuffer(buffer=shm.buf, dtype=arr.dtype).reshape(data_shape)

                self.shms.append(shm)
                in_memory_arrays[key] = shared_array
            else:
                in_memory_arrays[key] = np.zeros(data_shape, dtype=arr.dtype)

        self.cache_arrays = in_memory_arrays
        self.is_initialized = True
    
    def __getitem__(self, item):
        if self.is_item_cached[item]:
            return {k: v[item] for k, v in self.cache_arrays.items()}
        
        arrays = self.d.read_from_disk(item)
        arrays = self.d.precompose_data(arrays)
        for k, array in arrays.items():
            self.cache_arrays[k][item] = array
        self.is_item_cached[item] = True
        return arrays
        
    
    def __del__(self):
        if self.is_initialized and _get_rank()==0:
            for shm in self.shms:
                try:
                    shm.close()
                except Exception as e:
                    logging.error(f"Error when closing shared memory: {e}")
                try:
                    shm.unlink()
                except Exception as e:
                    logging.error(f"Error when unlinking shared memory: {e}")
        
        self.is_initialized = False
        self._is_first_process = False
    
    def remap(self, old_key: str, new_key: str):
        if self.cache_arrays is not None:
            self.cache_arrays[new_key] = self.cache_arrays.pop(old_key)
            