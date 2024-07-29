# utils/aws_scripts.py

import requests
from requests.auth import HTTPBasicAuth
import tqdm
import boto3
from boto3.s3.transfer import TransferConfig
import os
import pandas as pd
from utils.inference_scripts import perform_inf

from utils.custom_models import Resnet50_species, ResNet50_order, load_models


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

def download_object(s3_client, bucket_name, key, download_path, perform_inference=False,
                    remove_image=False, localisation_model=None, binary_model=None,
                    order_model=None, order_labels=None, species_model=None,
                    species_labels=None, country='UK', region='UKCEH', device=None,
                    order_data_thresholds=None, csv_file='results.csv'):
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
        s3_client.download_file(
            bucket_name, key, download_path, Config=transfer_config
        )
        if perform_inference:
            perform_inf(download_path, loc_model=localisation_model, binary_model=binary_model,
                        order_model=order_model, order_labels=order_labels, regional_model=species_model,
                        regional_category_map=species_labels, country=country, region=region, device=device,
                        order_data_thresholds=order_data_thresholds, csv_file=csv_file)
        if remove_image:
            os.remove(download_path)
    except Exception as e:
        print(f"Error downloading {bucket_name}/{key}: {e}")

def download_batch(s3_client, bucket_name, keys, local_path, perform_inference=False,
                   remove_image=False, localisation_model=None, binary_model=None,
                   order_model=None, order_labels=None, species_model=None, species_labels=None,
                   country='UK', region='UKCEH', device=None, order_data_thresholds=None, csv_file='results.csv'):
    """
    Download a batch of objects from S3.
    """
    for key in keys:
        file_path, filename = os.path.split(key)
        os.makedirs(os.path.join(local_path, file_path), exist_ok=True)
        download_path = os.path.join(local_path, file_path, filename)
        download_object(s3_client, bucket_name, key, download_path,
                        perform_inference, remove_image,
                        localisation_model, binary_model,
                        order_model, order_labels, species_model, species_labels,
                        country, region, device, order_data_thresholds, csv_file)

def count_files(s3_client, bucket_name, prefix):
    """
    Count number of files for a given prefix.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": bucket_name, "Prefix": prefix}
    page_iterator = paginator.paginate(**operation_parameters)

    count = 0
    for page in page_iterator:
        count += page.get("KeyCount", 0)

    return count

def get_objects(session, aws_credentials, bucket_name, key,
                local_path,
                batch_size=100,
                perform_inference=False,
                remove_image=False,
                localisation_model=None,
                binary_model=None,
                order_model=None, order_labels=None, species_model=None, species_labels=None,
               country='UK', region='UKCEH', device=None, order_data_thresholds=None, csv_file='results.csv'):
    """
    Fetch objects from the S3 bucket and download them synchronously in batches.
    """
    s3_client = session.client("s3", endpoint_url=aws_credentials["AWS_URL_ENDPOINT"])

    total_files = count_files(s3_client, bucket_name, key)

    paginator = s3_client.get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": bucket_name, "Prefix": key}
    page_iterator = paginator.paginate(**operation_parameters)

    progress_bar = tqdm.tqdm(total=total_files, desc="Download files from server synchronously")

    keys = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

            if len(keys) >= batch_size:
                download_batch(s3_client, bucket_name, keys, local_path, perform_inference,
                               remove_image, localisation_model, binary_model,
                               order_model, order_labels, species_model, species_labels,
                               country, region, device, order_data_thresholds, csv_file)
                keys = []
                progress_bar.update(batch_size)
        if keys:
            download_batch(s3_client, bucket_name, keys, local_path, perform_inference,
                           remove_image, localisation_model, binary_model,
                           order_model, order_labels, species_model, species_labels,
                           country, region, device, order_data_thresholds, csv_file)
            progress_bar.update(len(keys))

    progress_bar.close()
