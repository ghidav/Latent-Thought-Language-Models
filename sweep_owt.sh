#!/bin/bash
#SBATCH --job-name=ltm_sweep
#SBATCH --partition=ulow
#SBATCH --account=d.ghilardi
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=14-00:00:00
#SBATCH --output=sweep_owt_%j.log

set -euo pipefail

export DATA_CACHE_DIR="${DATA_CACHE_DIR:-/scratch_local/data_owt}"
export HF_HOME=/scratch_local/hf_cache

python sweep_owt.py "$@"
