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

echo "Downloading pre-tokenized OpenWebText into ${OUT_DIR} ..."

python -c "
from datasets import load_dataset
import numpy as np
import os

out_dir = '${OUT_DIR}'
os.makedirs(out_dir, exist_ok=True)

ds = load_dataset('anyasims/openwebtext-tokenized', split='train')
split = ds.train_test_split(test_size=0.005, seed=42)

for split_name, key in [('train', 'train'), ('val', 'test')]:
    total = 0
    with open(os.path.join(out_dir, f'{split_name}.bin'), 'wb') as f:
        for row in split[key]:
            arr = np.array(row['tokens'], dtype=np.uint16)
            f.write(arr.tobytes())
            total += len(arr)
    print(f'{split_name}: {total} tokens')
"

echo "Done. Files in ${OUT_DIR}:"
ls -lh "$OUT_DIR"
