import os
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm


def list_objects(session, bucket_name, prefix):
    """
    List all objects in an S3 bucket with a specific prefix.

    Args:
        session (boto3.Session): Authenticated AWS session.
        bucket_name (str): Name of the S3 bucket.
        prefix (str): Prefix for filtering objects.

    Returns:
        list: List of object keys in the bucket matching the prefix.
    """
    s3_client = session.client("s3")
    object_keys = []
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if "Contents" in page:
                object_keys.extend(obj["Key"] for obj in page["Contents"])
    except ClientError as e:
        print(f"\033[91mError listing objects: {e}\033[0m")
    return object_keys


def download_object(session, bucket_name, key, local_path, retries=3):
    """
    Download a single object from S3.

    Args:
        session (boto3.Session): Authenticated AWS session.
        bucket_name (str): S3 bucket name.
        key (str): Key of the object to download.
        local_path (str): Local directory to save the object.
        retries (int): Number of retries on failure.

    Returns:
        str: Local file path of the downloaded object.
    """
    s3_client = session.client("s3")
    local_file_path = os.path.join(local_path, key)
    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

    for attempt in range(retries):
        try:
            s3_client.download_file(bucket_name, key, local_file_path)
            return local_file_path
        except ClientError as e:
            print(f"\033[93mRetry {attempt + 1}/{retries} - Error downloading {key}: {e}\033[0m")
    raise RuntimeError(f"Failed to download {key} after {retries} attempts.")


def download_batch(session, bucket_name, keys, local_path, retries=3):
    """
    Download a batch of objects from S3.

    Args:
        session (boto3.Session): Authenticated AWS session.
        bucket_name (str): S3 bucket name.
        keys (list): List of object keys to download.
        local_path (str): Local directory to save objects.
        retries (int): Number of retries for each object.

    Returns:
        list: List of local file paths of successfully downloaded objects.
    """
    local_files = []
    for key in tqdm(keys, desc="Downloading batch"):
        try:
            local_file = download_object(session, bucket_name, key, local_path, retries)
            local_files.append(local_file)
        except RuntimeError as e:
            print(f"\033[91mSkipping {key}: {e}\033[0m")
    return local_files
