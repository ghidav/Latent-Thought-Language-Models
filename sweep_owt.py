"""
OWT training sweep across all 7 posterior optimizer configurations.

Usage:
    python sweep_owt.py                    # Run all 7 strategies sequentially
    python sweep_owt.py --dry-run          # Print commands without executing
    python sweep_owt.py --strategies 1 3 5 # Run only selected strategies (1-indexed)
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Sweep-level defaults (override via CLI args below)
# ---------------------------------------------------------------------------

MAX_ITERS = 10000
LR_DECAY_ITERS = 10000
WARMUP_ITERS = 500
EVAL_INTERVAL = 500
WANDB_PROJECT = "ltm-owt-sweep"

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

STRATEGIES = [
    {
        "name": "adamVI",
        "overrides": {
            "inference_method": "adamVI",
        },
    },
    {
        "name": "meta_learned",
        "overrides": {
            "inference_method": "meta_learned",
            "meta_hidden_dim": 128,
            "meta_bptt_depth": 4,
        },
    },
    {
        "name": "delta_momentum",
        "overrides": {
            "inference_method": "delta_momentum",
            "delta_alpha": 0.9,
        },
    },
    {
        "name": "precond_shampoo",
        "overrides": {
            "inference_method": "preconditioned_manifold",
            "precond_type": "shampoo",
            "precond_beta": 0.9,
        },
    },
    {
        "name": "precond_learned",
        "overrides": {
            "inference_method": "preconditioned_manifold",
            "precond_type": "learned",
            "precond_rank": 32,
        },
    },
    {
        "name": "langevin_fixed",
        "overrides": {
            "inference_method": "underdamped_langevin",
            "langevin_gamma": 0.1,
            "langevin_sigma": 0.01,
            "langevin_learn_schedule": False,
        },
    },
    {
        "name": "langevin_learned",
        "overrides": {
            "inference_method": "underdamped_langevin",
            "langevin_gamma": 0.1,
            "langevin_sigma": 0.01,
            "langevin_learn_schedule": True,
        },
    },
]


def build_cmd(strategy, group):
    """Build the command-line invocation for a single strategy run."""
    name = strategy["name"]
    overrides = strategy["overrides"]

    args = [
        sys.executable, "train_ltm.py",
        f"max_iters={MAX_ITERS}",
        f"lr_decay_iters={LR_DECAY_ITERS}",
        f"warmup_iters={WARMUP_ITERS}",
        f"eval_interval={EVAL_INTERVAL}",
        f"out_dir=output/sweep_{name}",
        f"wandb_log=True",
        f"wandb_project={WANDB_PROJECT}",
        f"wandb_run_name={name}",
        f"wandb_group={group}",
    ]

    for key, val in overrides.items():
        args.append(f"{key}={val}")

    return args


def main():
    parser = argparse.ArgumentParser(description="OWT sweep launcher")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--strategies", nargs="+", type=int, default=None,
                        help="1-indexed strategy numbers to run (e.g. --strategies 1 3 5)")
    args = parser.parse_args()

    if args.strategies:
        selected = [STRATEGIES[i - 1] for i in args.strategies]
    else:
        selected = STRATEGIES

    # Unique group name so all runs in this sweep are linked in wandb
    group = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"{'=' * 70}")
    print(f"OWT Sweep: {len(selected)} strategies, {MAX_ITERS} iters each")
    print(f"Eval every {EVAL_INTERVAL} iters | wandb project: {WANDB_PROJECT}")
    print(f"wandb group: {group}")
    print(f"{'=' * 70}")

    results = []

    for idx, strategy in enumerate(selected):
        cmd = build_cmd(strategy, group)
        name = strategy["name"]

        print(f"\n{'=' * 70}")
        print(f"[{idx + 1}/{len(selected)}] Starting: {name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'=' * 70}\n")

        if args.dry_run:
            results.append((name, "DRY-RUN", 0))
            continue

        t0 = time.time()
        proc = subprocess.run(cmd)
        elapsed = time.time() - t0

        status = "OK" if proc.returncode == 0 else f"FAIL (rc={proc.returncode})"
        results.append((name, status, elapsed))

        print(f"\n[{name}] {status} in {elapsed / 3600:.2f}h")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'Strategy':<25} {'Status':<15} {'Time (h)':<10}")
    print(f"{'-' * 70}")
    for name, status, elapsed in results:
        print(f"{name:<25} {status:<15} {elapsed / 3600:<10.2f}")
    print(f"{'=' * 70}")

    if any(s != "OK" and s != "DRY-RUN" for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
