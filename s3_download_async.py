#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script downloads files from an S3 bucket using asynchronous operations.
AWS credentials and S3 bucket name are loaded from a configuration file
(credentials.json).
"""

import sys
import os
import getpass
import json
import asyncio
import requests
from requests.auth import HTTPBasicAuth
import aioboto3
from boto3.s3.transfer import TransferConfig
import tqdm.asyncio


# Load AWS credentials and S3 bucket name from config file
with open("./credentials.json", encoding="utf-8") as config_file:
    aws_credentials = json.load(config_file)

# Initialize aioboto3 session
session = aioboto3.Session()

# Configure the transfer to optimize the download
transfer_config = TransferConfig(
    max_concurrency=20,  # Increase the number of concurrent transfers
    multipart_threshold=8 * 1024 * 1024,  # 8MB
    max_io_queue=1000,
    io_chunksize=262144,  # 256KB
)


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


async def download_object(s3_client, bucket_name, key, download_path):
    """
    Download a single object from S3 asynchronously.
    """
    try:
        await s3_client.download_file(
            bucket_name, key, download_path, Config=transfer_config
        )
    except Exception as e:
        print(f"Error downloading {bucket_name}/{key}: {e}")


async def download_batch(s3_client, bucket_name, keys, local_path):
    """
    Download a batch of objects from S3.
    """
    tasks = []
    for key in keys:
        file_path, filename = os.path.split(key)
        os.makedirs(os.path.join(local_path, file_path), exist_ok=True)
        download_path = os.path.join(local_path, file_path, filename)
        tasks.append(download_object(s3_client, bucket_name, key, download_path))
    await asyncio.gather(*tasks)


async def count_files(s3_client, bucket_name, prefix):
    """
    Count number of files for a given prefix.
    """
    paginator = s3_client.get_paginator("list_objects_v2")
    operation_parameters = {"Bucket": bucket_name, "Prefix": prefix}
    page_iterator = paginator.paginate(**operation_parameters)

    count = 0
    async for page in page_iterator:
        count += page.get("KeyCount", 0)

    return count


async def get_objects(bucket_name, key, local_path, batch_size=100):
    """
    Fetch objects from the S3 bucket and download them asynchronously in batches.
    """
    async with session.client(
        "s3",
        aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_credentials["AWS_REGION"],
        endpoint_url=aws_credentials["AWS_URL_ENDPOINT"],
    ) as s3_client:

        total_files = await count_files(s3_client, bucket_name, key)

        paginator = s3_client.get_paginator("list_objects_v2")
        operation_parameters = {"Bucket": bucket_name, "Prefix": key}
        page_iterator = paginator.paginate(**operation_parameters)

        progress_bar = tqdm.asyncio.tqdm(
            total=total_files, desc="Download files from server asynchronously."
        )

        keys = []
        async for page in page_iterator:
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
                if len(keys) >= batch_size:
                    await download_batch(s3_client, bucket_name, keys, local_path)
                    keys = []
                    progress_bar.update(batch_size)
            if keys:
                await download_batch(s3_client, bucket_name, keys, local_path)
                progress_bar.update(len(keys))

        progress_bar.close()


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def get_input(prompt):
    """Get user input."""
    return input(prompt + ": ")


def get_choice(prompt, options):
    """Get user's choice from a list of options."""
    print(prompt)
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")
    while True:
        choice = input("Choose an option (enter the number): ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Please try again.")


def display_menu():
    """Display the main menu and handle user interaction."""
    clear_screen()
    print("Download Files")
    print("============\n")

    username = get_input("API Username")
    password = getpass.getpass("API Password: ")

    all_deployments = get_deployments(username, password)

    countries = list({d["country"] for d in all_deployments if d["status"] == "active"})
    country = get_choice("Countries:", countries)

    country_deployments = [
        f"{d['location_name']} - {d['camera_id']}"
        for d in all_deployments
        if d["country"] == country and d["status"] == "active"
    ]
    deployment = get_choice("\nDeployments:", country_deployments)

    data_types = ["snapshot_images", "audible_recordings", "ultrasound_recordings"]
    data_type = get_choice("\nData type:", data_types)

    s3_bucket_name = [
        d["country_code"]
        for d in all_deployments
        if d["country"] == country and d["status"] == "active"
    ][0].lower()
    location_name, camera_id = deployment.split(" - ")
    dep_id = [
        d["deployment_id"]
        for d in all_deployments
        if d["country"] == country
        and d["location_name"] == location_name
        and d["camera_id"] == camera_id
        and d["status"] == "active"
    ][0]

    prefix = f"{dep_id}/{data_type}"

    print("\nSelect Directory:")
    while True:
        local_directory_path = get_input("Enter directory path")
        if os.path.isdir(local_directory_path):
            break
        print("Invalid directory. Please try again.")

    # Run the asynchronous download
    asyncio.run(get_objects(s3_bucket_name, prefix, local_directory_path))


if __name__ == "__main__":
    display_menu()
