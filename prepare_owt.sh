#!/bin/bash
#SBATCH --job-name=prepare_owt
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=prepare_owt_%j.log

set -euo pipefail

DATA_DIR="${DATA_CACHE_DIR:-/scratch_local/data_owt}"
OUT_DIR="${DATA_DIR}/tok_gpt2"

mkdir -p "$OUT_DIR"

echo "Downloading pre-tokenized OpenWebText into ${OUT_DIR} (streaming) ..."

python -c "
from datasets import load_dataset
import numpy as np
import os

out_dir = '${OUT_DIR}'
os.makedirs(out_dir, exist_ok=True)

ds = load_dataset('anyasims/openwebtext-tokenized', split='train', streaming=True)

train_f = open(os.path.join(out_dir, 'train.bin'), 'wb')
val_f = open(os.path.join(out_dir, 'val.bin'), 'wb')

train_total = 0
val_total = 0

# Use every 200th example as val (~0.5%)
for i, row in enumerate(ds):
    arr = np.array(row['ids'], dtype=np.uint16)
    if i % 200 == 0:
        val_f.write(arr.tobytes())
        val_total += len(arr)
    else:
        train_f.write(arr.tobytes())
        train_total += len(arr)
    if i % 500000 == 0:
        print(f'Processed {i} docs | train: {train_total} tokens, val: {val_total} tokens')

train_f.close()
val_f.close()
print(f'Done. train: {train_total} tokens, val: {val_total} tokens')
"

echo "Done. Files in ${OUT_DIR}:"
ls -lh "$OUT_DIR"
