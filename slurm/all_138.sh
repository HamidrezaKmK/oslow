#!/bin/bash

for i in {1..64}; do sbatch slurm/launch_64.slrm wandb agent --count 1 $@; sleep 0.5; done
for i in {1..32}; do sbatch slurm/launch_32.slrm  wandb agent --count 1 $@; sleep 0.5; done
for i in {1..16}; do sbatch slurm/launch_16.slrm  wandb agent --count 1 $@; sleep 0.5; done
# for i in {1..12}; do sbatch slurm/launch_12.slrm  wandb agent --count 1 $@; sleep 0.5; done
for i in {1..8}; do sbatch slurm/launch_8.slrm  wandb agent --count 1 $@; sleep 0.5; done
for i in {1..4}; do sbatch slurm/launch_4.slrm  wandb agent --count 1 $@; sleep 0.5; done
for i in {1..2}; do sbatch slurm/launch_2.slrm  wandb agent --count 1 $@; sleep 0.5; done
