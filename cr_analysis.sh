#!/bin/bash
#SBATCH --qos=turing
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --output=cr_train.out
#SBATCH --time=72:00:00          # total run time limit (DD-HH:MM:SS)

module purge; module load baskerville
module load bask-apps/live
module load CUDA/11.7.0
module load Miniconda3/4.10.3
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# Define the path to your existing Conda environment (modify as appropriate)
CONDA_ENV_PATH="/bask/projects/v/vjgo8416-amber/moth_detector_env/"

# Activate the environment
conda activate "${CONDA_ENV_PATH}"


# Run the Python script on baskerville
python s3_download_with_inference.py "Costa Rica" "All of the above"





