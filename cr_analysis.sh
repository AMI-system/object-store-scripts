#!/bin/bash
#SBATCH --output=cr_garden2.out

module purge; module load baskerville
module load bask-apps/live
module load CUDA/11.7.0
module load Miniconda3/4.10.3
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# Define the path to your existing Conda environment (modify as appropriate)
CONDA_ENV_PATH="~/moth_detector_env/"

# Activate the environment
conda activate "${CONDA_ENV_PATH}"

# Print the Costa Rica deployments avaialble on the object store
python print_deployments.py --subset_countries 'Costa Rica'

# Run the Python script on JASMIN
# python s3_download_with_inference.py \
#   --country "Costa Rica" \
#   --deployment "Garden - 3F1C4908"



