#!/bin/bash
#SBATCH --output=./logs/singapore.out
#SBATCH --time=24:00:00

source ~/miniforge3/bin/activate

# Define the path to your existing Conda environment (modify as appropriate)
CONDA_ENV_PATH="~/moth_detector_env/"

# Activate the environment
conda activate "${CONDA_ENV_PATH}"

# Print the Costa Rica deployments avaialble on the object store
# python print_deployments.py --subset_countries 'Costa Rica'

# Run the Inference script 
python s3_download_with_inference.py \
  --country "Singapore" \
  --deployment "All" \
  --crops_interval 10 \
  --keep_crops \
  --data_storage_path ./data/singapore

