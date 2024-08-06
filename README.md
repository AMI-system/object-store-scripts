# object-store-scripts

Directory to download images from Jasmin object store and perform inference to:
- detect objects
- classify objects as moth or non-moth
- identify the order
- determine the moth species

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the script, use the following command:

```bash
python s3_download_with_inference.py \
  --country "Costa Rica" \
  --deployment "Forest Edge - EC4AB109" \
  --keep_crops
```

Or to run on baskerville:

```bash
sbatch cr_analysis.sh
```


To download interactively, and without inference and subsequent image deletion:

```bash
python s3_download_async.py
```


To save the crops following inference run:

```
python create_crops_only.py
```