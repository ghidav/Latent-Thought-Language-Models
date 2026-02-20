"""
Download, preprocess and serve the OWT dataset as a DataLoader with latent variables.
"""

import argparse
import glob
import json
import os
import random
import re
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Base directory for caching dataset files
DATA_CACHE_DIR = os.environ.get("DATA_CACHE_DIR", "/scratch_local/data_owt")

# RNG seed parameters
DEFAULT_SEED = 42
RANK_SEED_OFFSET = 1337

# Latent variable initialization scale
LATENT_INIT_SCALE = 0.01

# -----------------------------------------------------------------------------
# Dataset Implementation
# -----------------------------------------------------------------------------

class PretokDatasetWithLatent(torch.utils.data.IterableDataset):
    """
    Loads pretokenized examples from disk and yields them as PyTorch tensors with latent variables.
    """

    def __init__(self, split, max_seq_len, max_z_len, z_dim):
        """
        Initialize the dataset.
        
        Args:
            split (str): Dataset split to use ('train' or 'val')
            max_seq_len (int): Maximum sequence length for tokens
            max_z_len (int): Maximum length of latent vectors
            z_dim (int): Dimension of latent vectors
        """
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.max_z_len = max_z_len
        self.z_dim = z_dim

    def __iter__(self):
        """
        Iterator that yields tokenized examples with latent variables.
        
        Yields:
            tuple: (x, y, z) where:
                - x is the input token sequence
                - y is the target token sequence (shifted by 1)
                - z is the latent variable vector
        """
        # Set up worker-specific RNG
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        
        # Get distributed training rank if applicable
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Create a unique seed based on worker and rank
        seed = DEFAULT_SEED + worker_id + RANK_SEED_OFFSET * rank
        rng = random.Random(seed)
        current_shard = None
        print(f"Created a PretokDatasetWithLatent with rng seed {seed}")

        # Find all available data shards
        bin_dir = os.path.join(DATA_CACHE_DIR, "tok_gpt2")
        shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))

        # Filter shards by split type
        if self.split == 'train':
            shard_filenames = [f for f in shard_filenames if 'train' in f]
        else:
            shard_filenames = [f for f in shard_filenames if 'val' in f]
            
        assert len(shard_filenames) > 0, f"No bin files found in {bin_dir}"
        
        # Infinite iteration for training
        while True:
            # Shuffle the order of shards
            rng.shuffle(shard_filenames)
            
            # Process each shard
            for shard in shard_filenames:
                if shard != current_shard:
                    current_shard = shard
                    print(f"Switching to new shard: {shard}")

                # Load the shard data using memory mapping
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                
                # Calculate number of complete batches in this shard
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "This shard is way too small. Please investigate."

                # Generate random latent variables for all batches in this shard
                z_matrix = np.random.randn(
                    num_batches, 
                    self.z_dim * self.max_z_len
                ).astype(np.float32) * LATENT_INIT_SCALE
                
                # Shuffle batch indices for randomness
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                
                # Process each batch
                for ix in ixs:
                    # Extract token sequence
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    
                    # Convert to PyTorch tensor and move to appropriate data type
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    
                    # Create input and target sequences (shifted by one token)
                    x = chunk[:-1]  # Input tokens
                    y = chunk[1:]   # Target tokens (next token prediction)
                    
                    # Get latent vector for this batch
                    z = z_matrix[ix]
                    
                    # Yield the example
                    yield x, y, z
    
    @staticmethod
    def get_shard_id(shard_file_name):
        # Extract shard number from filename (e.g., "train26.bin" → 26)
        match = re.search(r'(train|val)(\d+)\.bin', shard_file_name)
        if match:
            return int(match.group(2))
        return None


# -----------------------------------------------------------------------------
# Public Interface
# -----------------------------------------------------------------------------

class Task:
    """Task interface for working with the OWT dataset."""
    
    @staticmethod
    def iter_batches_with_latents(split, batch_size, max_seq_len, max_z_len, z_dim, 
                                  device, num_workers=0):
        """
        Create an iterable over batches of data with latent variables.
        
        Args:
            split (str): Dataset split ('train' or 'val')
            batch_size (int): Batch size
            max_seq_len (int): Maximum sequence length
            max_z_len (int): Maximum latent sequence length
            z_dim (int): Dimension of latent variables
            device (str): Device to load tensors to
            num_workers (int): Number of DataLoader workers
            
        Yields:
            tuple: (x, y, z) containing batches of:
                - x: Input token tensors [batch_size, max_seq_len]
                - y: Target token tensors [batch_size, max_seq_len]
                - z: Latent variable tensors [batch_size, z_dim * max_z_len]
        """
        # Create dataset instance
        ds = PretokDatasetWithLatent(
            split=split,
            max_seq_len=max_seq_len,
            max_z_len=max_z_len,
            z_dim=z_dim
        )
        
        # Create DataLoader
        dl = torch.utils.data.DataLoader(
            ds, 
            batch_size=batch_size, 
            pin_memory=True, 
            num_workers=num_workers
        )
        
        # Process and yield batches
        for x, y, z in dl:
            # Move tensors to requested device
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            
            # Yield the batch
            yield x, y, z
