#!/bin/bash
source venv/bin/activate
python plackett_luce.py wandb.run_name=${SLURM_JOB_ID} "$@"  # Since list can be in quotes, we need to pass "$@" instead of $@
