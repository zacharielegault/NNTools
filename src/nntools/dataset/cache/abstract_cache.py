from multiprocessing import shared_memory

import numpy as np


class AbstractCache:
    def __init__(self, dataset):
        self.d = dataset
        self.is_initialized = False
        self.id = self.d.id
        self.shms = []

    def init_shared_items_tracking(self):
        try:  # This is the wrong way to handle to do, we should check if we are at rank 0
            shm = shared_memory.SharedMemory(
                name=f"nntools_{self.id}_is_item_cached", size=self.nb_samples, create=True
            )
            self._is_first_process = True
        except FileExistsError:
            shm = shared_memory.SharedMemory(
                name=f"nntools_{self.id}_is_item_cached", size=self.nb_samples, create=False
            )
            self._is_first_process = False

        self.shms.append(shm)
        self._cache_items = np.frombuffer(buffer=shm.buf, dtype=bool)
        self._cache_items[:] = 0
    
    def remap(self, old_key: str, new_key: str):
        pass

    @property
    def nb_samples(self):
        return self.d.real_length