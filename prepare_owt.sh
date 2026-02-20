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

echo "Tokenizing OpenWebText into ${OUT_DIR} ..."

python -c "
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import os

out_dir = '${OUT_DIR}'
os.makedirs(out_dir, exist_ok=True)

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
dataset = load_dataset('openwebtext', trust_remote_code=True)

split = dataset['train'].train_test_split(test_size=0.005, seed=42)

for split_name, key in [('train', 'train'), ('val', 'test')]:
    all_tokens = []
    for example in tqdm(split[key], desc=f'Tokenizing {split_name}'):
        tokens = tokenizer.encode(example['text'])
        all_tokens.extend(tokens)
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(os.path.join(out_dir, f'{split_name}.bin'))
    print(f'{split_name}: {len(arr)} tokens')
"

echo "Done. Files in ${OUT_DIR}:"
ls -lh "$OUT_DIR"
