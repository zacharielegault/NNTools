import logging
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

from nntools.dataset.cache.abstract_cache import AbstractCache


class MemoryCache(AbstractCache):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.shms = []
        self.cache_with_shared_array = True
        self.cache_arrays = None
    
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

        shared_arrays = dict()
        nb_samples = self.d.real_length
        # Keep reference to all shm avoid the call from the garbage collector which pointer to buffer error
        if self.cache_with_shared_array:
            self.init_shared_items_tracking()

        for key, arr in arrays.items():
            if not isinstance(arr, np.ndarray):
                shared_arrays[key] = np.ndarray(nb_samples, dtype=type(arr))
                continue

            memory_shape = (nb_samples, *arr.shape)
            if self.cache_with_shared_array:
                try:
                    shm = shared_memory.SharedMemory(
                        name=f"nntools_{key}_{self.id}", size=arr.nbytes * nb_samples, create=True
                    )
                    logging.info(f"Creating shared memory, {mp.current_process().name}")
                    logging.debug(f"nntools_{key}_{self.id}: size: {shm.buf.nbytes} ({memory_shape})")
                except FileExistsError:
                    shm = shared_memory.SharedMemory(
                        name=f"nntools_{key}_{self.id}", size=arr.nbytes * nb_samples, create=False
                    )
                    logging.info(f"Accessing existing shared memory {mp.current_process().name}")

                self.shms.append(shm)

                shared_array = np.frombuffer(buffer=shm.buf, dtype=arr.dtype).reshape(memory_shape)

                shared_array[:] = 0
                # The initialization with 0 is not needed.
                # However, it's a good way to check if the shared memory is correctly initialized
                # It checks if there is enough space in dev/shm

                shared_arrays[key] = shared_array
            else:
                shared_arrays[key] = np.zeros(memory_shape, dtype=arr.dtype)

        self.cache_arrays = shared_arrays
        self.is_initialized = True
    
    def __getitem__(self, item):
        if self._cache_items[item]:
            return {k: v[item] for k, v in self.cache_arrays.items()}
        
        arrays = self.d.read_from_disk(item)
        arrays = self.d.precompose_data(arrays)
        for k, array in arrays.items():
            if array.ndim == 2:
                self.cache_arrays[k][item] = array
            else:
                self.cache_arrays[k][item] = array
        self._cache_items[item] = True
        return arrays
        
    
    def __del__(self):
        if self.is_initialized and self._is_first_process:
            for shm in self.shms:
                shm.close()
            if self._is_first_process:
                for shm in self.shms:
                    shm.unlink()
            self.is_initialized = False
            self._is_first_process = False
    
    def remap(self, old_key: str, new_key: str):
        if self.cache_arrays is not None:
            self.cache_arrays[new_key] = self.cache_arrays.pop(old_key)
            