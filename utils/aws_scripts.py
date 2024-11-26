# utils/aws_scripts.py

import requests
from requests.auth import HTTPBasicAuth
import tqdm
from boto3.s3.transfer import TransferConfig
import os
from datetime import datetime, timedelta
from utils.inference_scripts import perform_inf
import pandas as pd
import sys
import multiprocessing
import tqdm
import math
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import re


def get_deployments(username, password):
    """Fetch deployments from the API with authentication."""

    try:
        url = "https://connect-apps.ceh.ac.uk/ami-data-upload/get-deployments/"
        response = requests.get(
            url, auth=HTTPBasicAuth(username, password), timeout=600
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
        if response.status_code == 401:
            print("Wrong username or password. Try again!")
        sys.exit(1)
    except Exception as err:
        print(f"Error: {err}")
        sys.exit(1)


def download_object(
    s3_client,
    bucket_name,
    key,
    download_path,
    perform_inference=False,
    remove_image=False,
    localisation_model=None,
    binary_model=None,
    order_model=None,
    order_labels=None,
    country="UK",
    region="UKCEH",
    device=None,
    order_data_thresholds=None,
    csv_file="results.csv",
):
    """
    Download a single object from S3 synchronously.
    """

    # Configure the transfer to optimize the download
    transfer_config = TransferConfig(
        max_concurrency=20,  # Increase the number of concurrent transfers
        multipart_threshold=8 * 1024 * 1024,  # 8MB
        max_io_queue=1000,
        io_chunksize=262144,  # 256KB
    )

    try:
        s3_client.download_file(bucket_name, key, download_path, Config=transfer_config)

        # If crops are saved, define the frequecy
        
        save_crops = True
        print(f" - Saving crops for: {os.path.basename(download_path)}")

        if perform_inference:
            perform_inf(
                download_path,
                bucket_name=bucket_name,
                loc_model=localisation_model,
                binary_model=binary_model,
                order_model=order_model,
                order_labels=order_labels,
                country=country,
                region=region,
                device=device,
                order_data_thresholds=order_data_thresholds,
                csv_file=csv_file,
                save_crops=save_crops,
            )
        if remove_image:
            os.remove(download_path)
    except Exception as e:
        print(
            f"\033[91m\033[1m Error downloading {bucket_name}/{key}: {e}\033[0m\033[0m"
        )


def get_datetime_from_string(input):
    in_sting = input.replace("-snapshot.jpg", "")
    dt = in_sting.split("-")[-1]
    dt = datetime.strptime(dt, "%Y%m%d%H%M%S")
    return dt


def download_batch(
    s3_client,
    bucket_name,
    keys,
    local_path,
    perform_inference=False,
    remove_image=False,
    localisation_model=None,
    binary_model=None,
    order_model=None,
    order_labels=None,
    country="UK",
    region="UKCEH",
    device=None,
    order_data_thresholds=None,
    csv_file="results.csv",
    rerun_existing=False,
):
    """
    Download a batch of objects from S3.
    """

    existing_df = pd.read_csv(csv_file, dtype="unicode")

    for key in keys:
        file_path, filename = os.path.split(key)

        os.makedirs(os.path.join(local_path, file_path), exist_ok=True)
        download_path = os.path.join(local_path, file_path, filename)

        # check if file is in csv_file 'path' column
        if not rerun_existing:
            if existing_df["image_path"].str.contains(download_path).any():
                print(
                    f"{os.path.basename(download_path)} has already been processed. Skipping..."
                )
                continue

        print(key)
        download_object(
            s3_client,
            bucket_name,
            key,
            download_path,
            perform_inference,
            remove_image,
            localisation_model,
            binary_model,
            order_model,
            order_labels,
            country,
            region,
            device,
            order_data_thresholds,
            csv_file,
        )


def count_files(s3_client, bucket_name, prefix):
    """
    Count number of files for a given prefix.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": bucket_name, "Prefix": prefix}
    page_iterator = paginator.paginate(**operation_parameters)

    count = 0
    all_keys = []
    for page in page_iterator:
        if not os.path.basename(page.get("Contents", [])[0]["Key"]).startswith("$"):
            count += page.get("KeyCount", 0)
            file_i = page.get("Contents", [])[0]["Key"]
            all_keys = all_keys + [file_i]
    return count, all_keys

def get_objects(
    session,
    aws_credentials,
    bucket_name,
    prefix,
    local_path,
    batch_size=100,
    perform_inference=False,
    remove_image=False,
    localisation_model=None,
    binary_model=None,
    order_model=None,
    order_labels=None,
    country="UK",
    region="UKCEH",
    device=None,
    order_data_thresholds=None,
    csv_file="results.csv",
    rerun_existing=False,
    num_workers=1,
):
    """
    Fetch objects from the S3 bucket and download them synchronously or using threads.
    """
    s3_client = session.client("s3", endpoint_url=aws_credentials["AWS_URL_ENDPOINT"])

    total_files, all_keys = count_files(s3_client, bucket_name, prefix)

    paginator = s3_client.get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": bucket_name, "Prefix": prefix}
    page_iterator = paginator.paginate(**operation_parameters)

    keys = []
    for page in page_iterator:
        if os.path.basename(page.get("Contents", [])[0]["Key"]).startswith("$"):
            print(f'{page.get("Contents", [])[0]["Key"]} is suspected corrupt, skipping')
            continue

        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    # don't rerun previously analysed images
    results_df = pd.read_csv(csv_file, dtype=str)
    run_images = [re.sub(r'^.*?dep', 'dep', x) for x in results_df['image_path']]
    keys = [x for x in keys if x not in run_images]
    
    # Divide the keys among workers
    chunks = [
        keys[i : i + math.ceil(len(keys) / num_workers)]
        for i in range(0, len(keys), math.ceil(len(keys) / num_workers))
    ]

    # Shared progress bar
    progress_bar = tqdm.tqdm(total=total_files, desc=f"Download files for {os.path.basename(csv_file).replace('_results.csv', '')}")

    def process_chunk(chunk):
        for i in range(0, len(chunk), batch_size):
            batch_keys = chunk[i : i + batch_size]
            download_batch(
                s3_client,
                bucket_name,
                batch_keys,
                local_path,
                perform_inference,
                remove_image,
                localisation_model,
                binary_model,
                order_model,
                order_labels,
                country,
                region,
                device,
                order_data_thresholds,
                csv_file,
                rerun_existing,
            )
            progress_bar.update(len(batch_keys))

    # Use ThreadPoolExecutor instead of multiprocessing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_chunk, chunks)

    progress_bar.close()
