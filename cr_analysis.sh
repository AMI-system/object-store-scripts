#!/bin/bash
#SBATCH --output=./logs/cr_garden2.out

source ~/miniforge3/bin/activate

# Define the path to your existing Conda environment (modify as appropriate)
CONDA_ENV_PATH="~/moth_detector_env/"

# Activate the environment
conda activate "${CONDA_ENV_PATH}"

# Print the Costa Rica deployments avaialble on the object store
# python print_deployments.py --subset_countries 'Costa Rica'

# Run the Inference script 
python s3_download_with_inference.py \
  --country "Costa Rica" \
  --deployment "Garden - 3F1C4908"



