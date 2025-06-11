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
    # print(obj.key)
    obj.delete()
    return True

##################
# BUCKET = s3_client.Bucket("test-upload")
BUCKET = s3_client.Bucket("saltmarsh-soundscapes") # to clear test, use: test-upload
PREFIX = "dep000110" # deployment_id/data_type - to clear test, use: dep_test
##################

# Create a ThreadPoolExecutor
COUNT = 0
# for _ in BUCKET.objects.all():
for _ in BUCKET.objects.filter(Prefix=PREFIX):
    COUNT = COUNT + 1

with tqdm(desc="Deleting files", ncols=60, total=COUNT, unit="B", unit_scale=1) as pbar:
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(delete_file_from_bucket, obj)
            # for obj in BUCKET.objects.all(Prefix=PREFIX)
            for obj in BUCKET.objects.filter(Prefix=PREFIX)
        ]
        for _ in as_completed(futures):
            pbar.update(1)
