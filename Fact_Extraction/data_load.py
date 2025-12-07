import torch
import os
import logging
import logging_loader
from torch.utils.data import Dataset
import bisect

logger=logging.getLogger("Fact_Extraction_Model")

class TripleExtractionDataset(Dataset):
    def __init__(self, dir_filepath):
        self.chunk_paths = []
        self.chunk_sizes = []

        for root, dirs, files in os.walk(dir_filepath):
            for filename in sorted(files):
                if filename.endswith(".pt"):
                    path = os.path.join(root, filename)
                    self.chunk_paths.append(path)

                    # load only size
                    chunk = torch.load(path, map_location="cpu")
                    self.chunk_sizes.append(len(chunk))
        logger.info(f"Loaded {len(self.chunk_sizes)}chunks")

        # cumulative boundaries
        self.cum = []
        total = 0
        for s in self.chunk_sizes:
            total += s
            self.cum.append(total)
        logger.info(f"Loaded {len(self.cum)} Cumulative Chunks")

        # cache
        self._cached_chunk_idx = None
        self._cached_chunk = None

    def __len__(self):
        return self.cum[-1]

    def _load_chunk(self, idx):
        """Loads chunk only when needed, caches it."""
        if self._cached_chunk_idx == idx:
            return self._cached_chunk

        chunk = torch.load(self.chunk_paths[idx], map_location="cpu")
        self._cached_chunk = chunk
        self._cached_chunk_idx = idx
        logger.info(f"Chunk cached at {idx}")
        return chunk

    def __getitem__(self, idx):
        # find chunk index using binary search (FAST)
        chunk_idx = bisect.bisect_right(self.cum, idx)

        # local index inside chunk
        start = 0 if chunk_idx == 0 else self.cum[chunk_idx - 1]
        inside_idx = idx - start

        # get cached or load chunk
        chunk = self._load_chunk(chunk_idx)

        return chunk[inside_idx]
