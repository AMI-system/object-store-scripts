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
    species_model=None,
    species_labels=None,
    country="UK",
    region="UKCEH",
    device=None,
    order_data_thresholds=None,
    csv_file="results.csv",
    intervals=None,
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
        if intervals:
            path_df = get_datetime_from_string(os.path.basename(download_path))
            save_crops = path_df in intervals
        else:
            save_crops = False
        if save_crops:
            print(f" - Saving crops for: {os.path.basename(download_path)}")

        if perform_inference:
            perform_inf(
                download_path,
                bucket_name=bucket_name,
                loc_model=localisation_model,
                binary_model=binary_model,
                order_model=order_model,
                order_labels=order_labels,
                regional_model=species_model,
                regional_category_map=species_labels,
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
    species_model=None,
    species_labels=None,
    country="UK",
    region="UKCEH",
    device=None,
    order_data_thresholds=None,
    csv_file="results.csv",
    rerun_existing=False,
    intervals=None,
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
            species_model,
            species_labels,
            country,
            region,
            device,
            order_data_thresholds,
            csv_file,
            intervals,
        )


def count_files(s3_client, bucket_name, prefix):
    """
    Count number of files for a given prefix.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": bucket_name, "Prefix": prefix}
    page_iterator = paginator.paginate(**operation_parameters)

    count = 0
    first_page = ""
    last_page = ""
    for page in page_iterator:
        if count == 0:
            first_page = page.get("Contents", [])[0]["Key"]
        count += page.get("KeyCount", 0)
        last_page = page.get("Contents", [])[0]["Key"]
    return count, first_page, last_page


def get_objects(
    session,
    aws_credentials,
    bucket_name,
    key,
    local_path,
    batch_size=100,
    perform_inference=False,
    remove_image=False,
    localisation_model=None,
    binary_model=None,
    order_model=None,
    order_labels=None,
    species_model=None,
    species_labels=None,
    country="UK",
    region="UKCEH",
    device=None,
    order_data_thresholds=None,
    csv_file="results.csv",
    rerun_existing=False,
    crops_interval=None,
):
    """
    Fetch objects from the S3 bucket and download them synchronously in batches.
    """
    s3_client = session.client("s3", endpoint_url=aws_credentials["AWS_URL_ENDPOINT"])

    total_files, first_dt, last_dt = count_files(s3_client, bucket_name, key)

    paginator = s3_client.get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": bucket_name, "Prefix": key}
    page_iterator = paginator.paginate(**operation_parameters)

    progress_bar = tqdm.tqdm(
        total=total_files, desc="Download files from server synchronously"
    )

    if crops_interval is not None:
        first_dt = get_datetime_from_string(first_dt)
        last_dt = get_datetime_from_string(last_dt)
        t = first_dt
        intervals = []
        while t < last_dt:
            intervals = intervals + [t]
            t = t + timedelta(minutes=crops_interval)
    else:
        intervals = None

    keys = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

            if len(keys) >= batch_size:
                download_batch(
                    s3_client,
                    bucket_name,
                    keys,
                    local_path,
                    perform_inference,
                    remove_image,
                    localisation_model,
                    binary_model,
                    order_model,
                    order_labels,
                    species_model,
                    species_labels,
                    country,
                    region,
                    device,
                    order_data_thresholds,
                    csv_file,
                    rerun_existing,
                    intervals,
                )
                keys = []
                progress_bar.update(batch_size)
        if keys:
            download_batch(
                s3_client,
                bucket_name,
                keys,
                local_path,
                perform_inference,
                remove_image,
                localisation_model,
                binary_model,
                order_model,
                order_labels,
                species_model,
                species_labels,
                country,
                region,
                device,
                order_data_thresholds,
                csv_file,
                rerun_existing,
                intervals,
            )
            progress_bar.update(len(keys))

    progress_bar.close()
