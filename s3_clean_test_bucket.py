#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script cleans the test bucket.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm
import boto3


# Load AWS credentials and S3 bucket name from config file
with open("credentials.json", encoding="utf-8") as config_file:
    aws_credentials = json.load(config_file)

# Initialize S3 client
s3_client = boto3.resource(
    "s3",
    aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
    endpoint_url=aws_credentials["AWS_URL_ENDPOINT"],
    region_name=aws_credentials["AWS_REGION"],
)


def delete_file_from_bucket(obj):
    """Delete object"""
    obj.delete()
    return True


BUCKET = s3_client.Bucket("test-upload")
PREFIX = ""

COUNT = 0

for i in BUCKET.objects.filter(Prefix=PREFIX):
    COUNT = COUNT + 1

print(f"Total number of files {COUNT}\n")

# Create a ThreadPoolExecutor
with tqdm(desc="Deleting files", ncols=60, total=COUNT, unit="B", unit_scale=1) as pbar:
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(delete_file_from_bucket, obj)
            for obj in BUCKET.objects.filter(Prefix=PREFIX)
        ]
        for future in as_completed(futures):
            pbar.update(1)
