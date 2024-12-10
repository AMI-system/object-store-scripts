import argparse
import boto3
import json
import os
from boto3.s3.transfer import TransferConfig
import torch

from utils.custom_models import load_models
from utils.inference_scripts import perform_inf


# Transfer configuration for optimised S3 download
transfer_config = TransferConfig(
    max_concurrency=20,  # Increase the number of concurrent transfers
    multipart_threshold=8 * 1024 * 1024,  # 8MB
    max_io_queue=1000,
    io_chunksize=262144,  # 256KB
)


def initialise_session(credentials_file="credentials.json"):
    """
    Load AWS and API credentials from a configuration file and initialise an AWS session.

    Args:
        credentials_file (str): Path to the credentials JSON file.

    Returns:
        boto3.Client: Initialised S3 client.
    """
    with open(credentials_file, encoding="utf-8") as config_file:
        aws_credentials = json.load(config_file)
    session = boto3.Session(
        aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_credentials["AWS_REGION"],
    )
    client = session.client("s3", endpoint_url=aws_credentials["AWS_URL_ENDPOINT"])
    return client


def download_and_analyse(
    keys,
    output_dir,
    bucket_name,
    client,
    remove_image=True,
    perform_inference=True,
    save_crops=False,
    localisation_model=None,
    box_threshold=0.99,
    binary_model=None,
    order_model=None,
    order_labels=None,
    species_model=None,
    species_labels=None,
    device=None,
    order_data_thresholds=None,
    csv_file="results.csv",
):
    """
    Download images from S3 and perform analysis.

    Args:
        keys (list): List of S3 keys to process.
        output_dir (str): Directory to save downloaded files and results.
        bucket_name (str): S3 bucket name.
        client (boto3.Client): Initialised S3 client.
        Other args: Parameters for inference and analysis.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for key in keys:
        local_path = os.path.join(output_dir, os.path.basename(key))
        print(f"Downloading {key} to {local_path}")
        client.download_file(bucket_name, key, local_path, Config=transfer_config)

        # Perform image analysis if enabled
        print(f"Analysing {local_path}")
        if perform_inference:
            perform_inf(
                local_path,
                bucket_name=bucket_name,
                loc_model=localisation_model,
                box_threshold=box_threshold,
                binary_model=binary_model,
                order_model=order_model,
                order_labels=order_labels,
                regional_model=species_model,
                regional_category_map=species_labels,
                proc_device=device,
                order_data_thresholds=order_data_thresholds,
                csv_file=csv_file,
                save_crops=save_crops,
            )
        # Remove the image if cleanup is enabled
        if remove_image:
            os.remove(local_path)


def main(
    chunk_id,
    json_file,
    output_dir,
    bucket_name,
    credentials_file="credentials.json",
    remove_image=True,
    perform_inference=True,
    save_crops=False,
    localisation_model=None,
    box_threshold=0.99,
    binary_model=None,
    order_model=None,
    order_labels=None,
    species_model=None,
    species_labels=None,
    device=None,
    order_data_thresholds=None,
    csv_file="results.csv",
):
    """
    Main function to process a specific chunk of S3 keys.

    Args:
        chunk_id (str): ID of the chunk to process (e.g., chunk_0).
        json_file (str): Path to the JSON file with key chunks.
        output_dir (str): Directory to save results.
        bucket_name (str): S3 bucket name.
        Other args: Parameters for download and analysis.
    """
    with open(json_file, "r") as f:
        chunks = json.load(f)

    if chunk_id not in chunks:
        raise ValueError(f"Chunk ID {chunk_id} not found in JSON file.")

    client = initialise_session(credentials_file)

    keys = chunks[chunk_id]["keys"]
    download_and_analyse(
        keys=keys,
        output_dir=output_dir,
        bucket_name=bucket_name,
        client=client,
        remove_image=remove_image,
        perform_inference=perform_inference,
        save_crops=save_crops,
        localisation_model=localisation_model,
        box_threshold=box_threshold,
        binary_model=binary_model,
        order_model=order_model,
        order_labels=order_labels,
        species_model=species_model,
        species_labels=species_labels,
        device=device,
        order_data_thresholds=order_data_thresholds,
        csv_file=csv_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a specific chunk of S3 keys.")
    parser.add_argument(
        "--chunk_id",
        required=True,
        help="ID of the chunk to process (e.g., 0, 1, 2, 3).",
    )
    parser.add_argument(
        "--json_file", required=True, help="Path to the JSON file with key chunks."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save downloaded files and analysis results.",
        default="./data/",
    )
    parser.add_argument("--bucket_name", required=True, help="Name of the S3 bucket.")
    parser.add_argument(
        "--credentials_file",
        default="credentials.json",
        help="Path to AWS credentials file.",
    )
    parser.add_argument(
        "--remove_image", action="store_true", help="Remove images after processing."
    )
    parser.add_argument(
        "--perform_inference", action="store_true", help="Enable inference."
    )
    parser.add_argument(
        "--save_crops", action="store_true", help="Whether to save the crops."
    )
    parser.add_argument(
        "--localisation_model_path",
        type=str,
        help="Path to the localisation model weights.",
        default="./models/v1_localizmodel_2021-08-17-12-06.pt",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.99,
        help="Threshold for the confidence score of bounding boxes. Default: 0.99",
    )
    parser.add_argument(
        "--binary_model_path",
        type=str,
        help="Path to the binary model weights.",
        default="./models/moth-nonmoth-effv2b3_20220506_061527_30.pth",
    )
    parser.add_argument(
        "--order_model_path",
        type=str,
        help="Path to the order model weights.",
        default="./models/dhc_best_128.pth",
    )
    parser.add_argument(
        "--order_labels", type=str, help="Path to the order labels file."
    )
    parser.add_argument(
        "--species_model_path",
        type=str,
        help="Path to the species model weights.",
        default="./models/turing-costarica_v03_resnet50_2024-06-04-16-17_state.pt",
    )
    parser.add_argument(
        "--species_labels",
        type=str,
        help="Path to the species labels file.",
        default="./models/03_costarica_data_category_map.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (e.g., cpu or cuda).",
    )
    parser.add_argument(
        "--order_thresholds_path",
        type=str,
        help="Path to the order data thresholds file.",
        default="./models/thresholdsTestTrain.csv",
    )
    parser.add_argument(
        "--csv_file", default="results.csv", help="Path to save analysis results."
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            "\033[95m\033[1mCuda available, using GPU "
            + "\N{White Heavy Check Mark}\033[0m\033[0m"
        )
    else:
        device = torch.device("cpu")
        print(
            "\033[95m\033[1mCuda not available, using CPU "
            + "\N{Cross Mark}\033[0m\033[0m"
        )

    models = load_models(
        device,
        args.localisation_model_path,
        args.binary_model_path,
        args.order_model_path,
        args.order_thresholds_path,
        args.species_model_path,
        args.species_labels,
    )

    main(
        chunk_id=args.chunk_id,
        json_file=args.json_file,
        output_dir=args.output_dir,
        bucket_name=args.bucket_name,
        credentials_file=args.credentials_file,
        remove_image=args.remove_image,
        save_crops=args.save_crops,
        perform_inference=args.perform_inference,
        localisation_model=models["localisation_model"],
        box_threshold=args.box_threshold,
        binary_model=models["classification_model"],
        order_model=models["order_model"],
        order_labels=models["order_model_labels"],
        order_data_thresholds=models["order_model_thresholds"],
        species_model=models["species_model"],
        species_labels=models["species_model_labels"],
        device=device,
        csv_file=args.csv_file,
    )
