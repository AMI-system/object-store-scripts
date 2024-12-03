#!/bin/bash

#SBATCH --job-name=generate_keys
#SBATCH --output=generate_keys.out
#SBATCH --time=01:00:00   
#SBATCH --mem=8G 
#SBATCH --partition=short-serial   


source ~/miniforge3/bin/activate
conda activate "~/conda_envs/moth_detector_env/"

###########################
# Step 2: generate the keys
###########################

# # Define deployment IDs
# pan_deployments=("dep000086" "dep000088" "dep000092" "dep000091" "dep000084" "dep000021" "dep000020" "dep000017" "dep000018" "dep000083" "dep000087" "dep000090" "dep000022" "dep000089")


# for deployment_id in "${pan_deployments[@]}"; do
#   echo "Processing deployment: $deployment_id"

#   python 02_generate_keys.py \
#     --bucket 'pan' \
#     --deployment_id "$deployment_id" \
#     --output_file "./keys/harlequin/pan/${deployment_id}_keys.txt"
# done

# # Define deployment IDs
# cr_deployments=("dep000036" "dep000031" "dep000035" "dep000038" "dep000034" "dep000033" "dep000037" "dep000039" "dep000032" "dep000040")


# for deployment_id in "${cr_deployments[@]}"; do
#   echo "Processing deployment: $deployment_id"

#   python 02_generate_keys.py \
#     --bucket 'cri' \
#     --deployment_id "$deployment_id" \
#     --output_file "./keys/harlequin/cri/${deployment_id}_keys.txt"
# done

###########################
# Step 3: split into chunks
###########################
input_directory="./keys/harlequin"

# Loop through each file in the subdirectory
for input_file in "$input_directory"/*/*_keys.txt; do
  deployment_id=$(basename "$input_file" | sed 's/_keys.txt//')
  output_file="${input_file/_keys.txt/_workload_chunks.json}"  

  echo "Processing file: $input_file"
  echo "Output file: $output_file"

  # Run the Python script with the appropriate arguments
  python 03_pre_chop_files.py \
    --input_file "$input_file" \
    --file_extensions 'jpg' 'jpeg' \
    --chunk_size 100 \
    --output_file "$output_file"

  # Check if the script executed successfully
  if [ $? -ne 0 ]; then
    echo "Error processing file: $input_file"
    exit 1
  fi
done






