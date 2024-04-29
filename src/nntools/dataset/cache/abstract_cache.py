from abc import abstractmethod
from multiprocessing import shared_memory
from torch.utils.data import DataLoader
import numpy as np
import tqdm

class AbstractCache:
    def __init__(self, dataset):
        self.d = dataset
        self.is_initialized = False
        self.id = self.d.id
        self.shms = []
        self._num_workers = 12
        self._batch_size = 32

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
    
    @abstractmethod
    def remap(self, old_key: str, new_key: str):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass
    
    @abstractmethod
    def init_cache(self):
        pass
    
    @property
    def nb_samples(self):
        return self.d.real_length
    
    def auto_cache(self, method='thread'):
        self.init_cache()
        dataloader = DataLoader(self.d, 
                                num_workers=self._num_workers, batch_size=self._batch_size, 
                                pin_memory=False, shuffle=False)
        for i in tqdm.tqdm(dataloader, total=len(dataloader)):
            pass