# Performing AMBER Inference on JASMIN

This directory is designed to download images from Jasmin object store and perform inference to:
- detect objects
- classify objects as moth or non-moth
- identify the order

This branch is designed to save crops of beetles. 

## JASMIN Set-Up

To use this pipeline on JASMIN you must have access to the following services: 
- **Login Services**: jasmin-login. This provides access to the JASMIN shared services, i.e. login, transfer, scientific analysis servers, Jupyter notebook and LOTUS.
- **Object Store**: ami-test-o. This is the data object store tenancy for the Automated Monitoring of Insects Trap.

The [JASMIN documentation](https://help.jasmin.ac.uk/docs/getting-started/get-started-with-jasmin/) provides useful infomration on how to get set-up with these services. Including: 
1. [Generate an SSH key](https://help.jasmin.ac.uk/docs/getting-started/generate-ssh-key-pair/)
2. [Getting a JASMIN portal account](https://help.jasmin.ac.uk/docs/getting-started/get-jasmin-portal-account/)
3. [Request “jasmin-login” access](https://help.jasmin.ac.uk/docs/getting-started/get-login-account/) (access to the shared JASMIN servers and the LOTUS batch cluster)

## Models

You will need to add the models files to the ./models subdirectory. Following this you can pass in: 
- binary_model_path: The path to the binary model weights 
- order_model_path: The path to the binary model weights
- order_threshold_path: The path to the binary model weights
- localisation_model_path: The path to the binary model weights

AMBER team members can find these files on [OneDrive](https://thealanturininstitute.sharepoint.com/:f:/r/sites/Automatedbiodiversitymonitoring/Shared%20Documents/General/Data/models/jasmin?csf=1&web=1&e=HgjhgA). Others can contact [Katriona Goldmann](kgoldmann@turing.ac.uk) for the model files. 


## Conda Environment and Installation

Once you have access to JASMIN, you will need to [install miniforge](https://help.jasmin.ac.uk/docs/software-on-jasmin/creating-and-using-miniforge-environments/) to run condat. Then create a conda environment and install packages: 

```bash
CONDA_ENV_PATH="~/conda_envs/moth_detector_env/"
source ~/miniforge3/bin/activate
conda create -p "${CONDA_ENV_PATH}" python=3.9
conda activate "${CONDA_ENV_PATH}"

conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install --yes --file requirements.txt
```

## Configs

To use the inference scripts you will need to set up a `credentials.json` file containing: 

```json
{
  "AWS_ACCESS_KEY_ID": `SECRET`,
  "AWS_SECRET_ACCESS_KEY": `SECRET`,
  "AWS_REGION": `SECRET`,
  "AWS_URL_ENDPOINT": `SECRET`,
  "UKCEH_username": `SECRET`,
  "UKCEH_password": `SECRET`,
  "directory": './inferences/data'
}
```

Contact [Katriona Goldmann](kgoldmann@turing.ac.uk) for the AWS Access and UKCEH API configs. 

## Usage

Load the conda env:

```bash
source ~/miniforge3/bin/activate
conda activate "~/conda_envs/moth_detector_env/"
```

<!-- Inferences are run by country and deployment site. To run the script, for Costa Rica say, use the following command:

```bash
python s3_download_with_inference.py \
  --country "Costa Rica" \
  --deployment "Forest Edge - EC4AB109" \
  --data_storage_path "./data/harlequin/CostaRica" \
  --num_workers 1
```

To run for all deployments use `--deployment "All"` -->

The multi-core pipeline is run in several steps: 

1. Listing All Available Deployments
2. Generate Key Files
3. Chop the keys into chunks
4. Analyse the chunks

### 01. Listing Available Deployments

To find information about the available deployments you can use the print_deployments function. For all deployments: 

```bash
python print_deployments.py --include_inactive
```

or for Costa Rica and Panama only: 

```bash
python print_deployments.py \
  --subset_countries 'Costa Rica' 
```

For deployments of interest you can then run `s3_download_with_inference.py`, as above, where the `--deployment` argument is passed as the deployment key value. e.g.: 

```bash
python s3_download_with_inference.py \
  --country "Costa Rica" \
  --deployment "Garden - 3F1C4908"
```

### 02. Generating the Keys

```bash
python 02_generate_keys.py --bucket 'cri' --deployment_id 'dep000031' --output_file './keys/dep000031_keys.txt'
```

### 03. Pre-chop the Keys into Chunks

```bash
python 03_pre_chop_files.py --input_file './keys/dep000031_keys.txt' --file_extension 'jpg|jpeg' --chunk_size 100 --output_file './keys/dep000031_workload_chunks.json'
```

### 04. Process the Chunked Files

```bash
python 04_process_chunks.py \
  --chunk_id 1 \
  --json_file './keys/dep000031_workload_chunks.json' \
  --output_dir './data/dep000031' \
  --bucket_name 'cri' \
  --credentials_file './credentials.json' \
  --perform_inference \
  --remove_image
```


## Running with slurm

To run with slurm you need to be logged in on the [scientific nodes](https://help.jasmin.ac.uk/docs/interactive-computing/sci-servers/). 

It is recommended you set up a shell script to runfor your country and deployment of interest. For example, `cr_analysis.sh` peformes inferences for Costa Rica's Garden - 3F1C4908 deployment. You can run this using: 

```bash
sbatch harlequin_costarica.sh
```

Note to run slurm you will need to install miniforge on the scientific nodes. 

To check the slurm queue: 

```bash
squeue -u USERNAME
```

