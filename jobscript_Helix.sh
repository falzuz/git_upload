#!/bin/bash

#SBATCH --job-name=dipolar-CL
#SBATCH --output=dipolar_%A_%a.out

#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:A100:1
#SBATCH --time=48:00:00
#SBATCH --mem=8gb

#SBATCH --array=0-19

module load devel/cuda/12.6
module load compiler/gnu/11.3

./bosegascl$SLURM_ARRAY_TASK_ID
