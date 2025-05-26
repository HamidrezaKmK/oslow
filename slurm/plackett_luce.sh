#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=ml
#SBATCH --qos=ml
#SBATCH --account=ml
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=concerto[1-3],overture

# Load the environment
. /mfs1/u/$USER/envs/oslow

# OPTIONS: 'sinusoid_affine' 'nonparametric_affine' 'sigmoid_affine' 'absolute_affine'
for data_type in 'pnl_sigmoid_sinusoid' 'pnl_sinusoid_softplus'; do
    python /h/319/aidanl/oslow/plackett_luce.py -m \
        data="$data_type" \
        data.graph_generator.graph_type='choice(erdos_renyi,full)' \
        data.graph_generator.num_nodes='range(3,11)' \
        seed='range(1,11)' # not inclusive of the last value
        
        # 'glob(*)'
        # 'choice(erdos_renyi,full)'
    wandb sync --clean --clean-old-hours 0
    sleep 30
done

