#!/bin/bash
#SBATCH --output=./logs/harlequin.out
#SBATCH --partition=par-single
#SBATCH --time=47:30:00

# see https://help.jasmin.ac.uk/docs/batch-computing/slurm-queues/ from slurm help

source ~/miniforge3/bin/activate

# Define the path to your existing Conda environment (modify as appropriate)
CONDA_ENV_PATH="~/conda_envs/moth_detector_env/"

# Activate the environment
conda activate "${CONDA_ENV_PATH}"

deps=("Old Growth - E601A49B" "Forest Edge - EC4AB109" "Pre-montane forest - 961AC5EB" "Garden - 3F1C4908")

for dep in "${deps[@]}"; do
    python s3_download_with_inference.py \
      --country "Costa Rica" \
      --deployment "$dep" \
      --keep_crops \
      --data_storage_path ./data/harlequin
done

python s3_download_with_inference.py \
      --country "Panama" \
      --deployment "All" \
      --keep_crops \
      --data_storage_path ./data/harlequin