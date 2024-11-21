#!/bin/bash
#SBATCH --output=./logs/harlequin.out
#SBATCH --time=24:00:00
#SBATCH --constraint="skylake348G"

# for help, see:
# - https://help.jasmin.ac.uk/docs/batch-computing/slurm-queues/
# - https://help.jasmin.ac.uk/docs/batch-computing/lotus-cluster-specification/
# - https://help.jasmin.ac.uk/docs/batch-computing/orchid-gpu-cluster/

source ~/miniforge3/bin/activate

# Define the path to your existing Conda environment (modify as appropriate)
CONDA_ENV_PATH="~/conda_envs/moth_detector_env/"

# Activate the environment
conda activate "${CONDA_ENV_PATH}"

deps=("Swamp - 5794FB2E" "Garden - 3F1C4908" "Old Growth - E601A49B" "Forest Edge - EC4AB109" "Pre-montane forest - 961AC5EB")

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