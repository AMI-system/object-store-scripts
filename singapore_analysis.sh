#!/bin/bash
#SBATCH --output=./logs/singapore.out
#SBATCH --partition=par-single
#SBATCH --time=47:30:00

# see https://help.jasmin.ac.uk/docs/batch-computing/slurm-queues/ from slurm help

source ~/miniforge3/bin/activate

# Define the path to your existing Conda environment (modify as appropriate)
CONDA_ENV_PATH="~/moth_detector_env/"

# Activate the environment
conda activate "${CONDA_ENV_PATH}"

# Print the Costa Rica deployments avaialble on the object store
# python print_deployments.py --subset_countries 'Thailand'

# Run the Inference script 
python s3_download_with_inference.py \
  --country "Singapore" \
  --deployment "All" \
  --crops_interval 10 \
  --keep_crops \
  --data_storage_path ./data/singapore

