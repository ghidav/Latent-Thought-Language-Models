#!/bin/bash
#SBATCH --job-name=prepare_owt
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=prepare_owt_%j.log

set -euo pipefail

DATA_DIR="${DATA_CACHE_DIR:-/scratch_local/data_owt}"
OUT_DIR="${DATA_DIR}/tok_gpt2"
NUM_PROC="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$OUT_DIR"

echo "Tokenizing OpenWebText into ${OUT_DIR} with ${NUM_PROC} workers ..."

python -c "
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import os

out_dir = '${OUT_DIR}'
num_proc = int('${NUM_PROC}')
os.makedirs(out_dir, exist_ok=True)

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
dataset = load_dataset('openwebtext', trust_remote_code=True)

split = dataset['train'].train_test_split(test_size=0.005, seed=42)

def tokenize(example):
    return {'tokens': tokenizer.encode(example['text'])}

for split_name, key in [('train', 'train'), ('val', 'test')]:
    tokenized = split[key].map(tokenize, num_proc=num_proc, remove_columns=['text'], desc=f'Tokenizing {split_name}')
    all_tokens = np.concatenate([np.array(t, dtype=np.uint16) for t in tokenized['tokens']])
    all_tokens.tofile(os.path.join(out_dir, f'{split_name}.bin'))
    print(f'{split_name}: {len(all_tokens)} tokens')
"

echo "Done. Files in ${OUT_DIR}:"
ls -lh "$OUT_DIR"
