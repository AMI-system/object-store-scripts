# utils/download_and_crop.py

import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from boto3.s3.transfer import TransferConfig
import boto3
import json


def download_image(url):
    """Download an image from a given URL."""
    try:
        response = requests.get(url, timeout=600)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image from {url}: {e}")
        return None


def crop_and_save_image(image, crop_box, save_path):
    """Crop and save the image to the specified path."""
    cropped_image = image.crop(crop_box)
    cropped_image.save(save_path)


def download_image_from_s3(s3_client, bucket_name, key, download_path):
    """Download a single image from S3."""
    transfer_config = TransferConfig(
        max_concurrency=20,
        multipart_threshold=8 * 1024 * 1024,
        max_io_queue=1000,
        io_chunksize=262144,
    )

    try:
        s3_client.download_file(bucket_name, key, download_path, Config=transfer_config)
    except Exception as e:
        print(f"Error downloading {bucket_name}/{key}: {e}")


def process_images_from_csv_s3(session, aws_credentials, csv_path, local_path):
    """Process images listed in a CSV file from S3."""
    s3_client = session.client("s3", endpoint_url=aws_credentials["AWS_URL_ENDPOINT"])
    df = pd.read_csv(csv_path)
    df["count"] = df.groupby(["image_path"]).cumcount()

    df["image_time"] = df["image_path"].str.split("/").str[-1]
    df["image_time"] = df["image_time"].str.split("-").str[1]
    df["image_time"] = pd.to_datetime(df["image_time"], format="%Y%m%d%H%M%S%f")

    # get all unique image_times
    image_times = df["image_time"].unique()

    # sort df by image_time
    df = df.sort_values(by="image_time")

    # get the first image_time
    first_image_time = image_times[0]
    last_image_time = image_times[-1]

    # get all 10 minute intervals between first and last image_time
    intervals = pd.date_range(start=first_image_time, end=last_image_time, freq="10T")
    df = df[df["image_time"].isin(intervals)]

    # remove all before snapshot from image_path
    df["crop_name"] = df["image_path"].str.split("/").str[-1]
    df["crop_name"] = (
        df["crop_name"].str.split(".").str[0]
        + "_crop"
        + df["count"].astype(str).str.zfill(2)
        + ".jpg"
    )

    # sort by crop_name
    df = df.sort_values(by="crop_name")

    # save the csv
    os.makedirs(local_path, exist_ok=True)
    df.to_csv(os.path.join(local_path, os.path.basename(csv_path)), index=False)

    print(f"Processing {len(df)} crops and {len(df['image_path'].unique())} images")

    for index, row in df.iterrows():
        xmin, ymin, xmax, ymax = row["x_min"], row["y_min"], row["x_max"], row["y_max"]

        # from row['image_path'] keep all after third last /
        key = "/".join(row["image_path"].split("/")[-3:])
        bucket_name = "cri"
        file_path, filename = os.path.split(key)

        save_file = row["crop_name"]
        save_dir = os.path.join(local_path, file_path)

        os.makedirs(save_dir, exist_ok=True)
        download_path = os.path.join(save_dir, filename)
        crop_path = os.path.join(save_dir, save_file)

        download_image_from_s3(s3_client, bucket_name, key, download_path)

        image = Image.open(download_path)

        # get the width of the image
        width, height = image.size
        os.remove(download_path)

        # skip if over half the image
        if xmax - xmin > width / 2 or ymax - ymin > height / 2:
            continue

        crop_box = (xmin, ymin, xmax, ymax)
        crop_and_save_image(image, crop_box, crop_path)


if __name__ == "__main__":
    csv_path = "/bask/homes/f/fspo1218/amber/projects/object-store-scripts/data/Costa_Rica_results.csv"  # Path to your CSV file
    local_path = "/bask/projects/v/vjgo8416-amber/projects/object-store-scripts/crops/"  # Local directory to save images

    # If downloading from S3
    with open("./credentials.json", encoding="utf-8") as config_file:
        aws_credentials = json.load(config_file)
    session = boto3.Session(
        aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_credentials["AWS_REGION"],
    )
    # aws_endpoint = {"AWS_URL_ENDPOINT": "https://s3.amazonaws.com"}
    process_images_from_csv_s3(session, aws_credentials, csv_path, local_path)
