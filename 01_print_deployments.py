#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script downloads files from an S3 bucket synchronously and performs
inference on the images. AWS credentials, S3 bucket name, and UKCEH API
credentials are loaded from a configuration file (credentials.json).
"""

import json
import argparse
import sys
import requests
import boto3
from requests.auth import HTTPBasicAuth


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


def initialise_session(credentials):
    """Initialise the AWS S3 client"""
    session = boto3.Session(
        aws_access_key_id=credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=credentials["AWS_REGION"],
    )
    s3_client = session.client("s3", endpoint_url=credentials["AWS_URL_ENDPOINT"])
    return s3_client


def deployment_image_counter(dep, client, country, print_count):
    """Calculate the number of images for a given deployment"""
    deployment_id = dep["deployment_id"]
    location_name = dep["location_name"]
    camera_id = dep["camera_id"]
    print(
        f" - Deployment ID: {deployment_id}, Name: {location_name}, Camera ID: {camera_id}"
    )

    if print_count:
        prefix = f"{deployment_id}/snapshot_images"
        image_count = count_files(client, country, prefix)
        print(f"   Images: {image_count}")

    return image_count


def print_deployments(
    credentials,
    include_inactive=False,
    subset_countries=None,
    print_image_count=True,
):
    """Print deployment details, optionally filtering by country or active status."""
    username, password = (
        credentials["UKCEH_username"],
        credentials["UKCEH_password"],
    )
    deployments = get_deployments(username, password)

    # Filter active deployments if not including inactive
    if not include_inactive:
        deployments = [d for d in deployments if d["status"] == "active"]

    # Initialise S3 client
    client = initialise_session(credentials)

    # Prepare subset of countries if specified
    all_countries = sorted({dep["country"].title() for dep in deployments})
    if subset_countries:
        subset_countries = [c.title() for c in subset_countries]
        missing_countries = [c for c in subset_countries if c not in all_countries]
        for missing in missing_countries:
            print(f"WARNING: No deployments found for '{missing}'.")
        all_countries = [c for c in all_countries if c in subset_countries]

    # Print deployments for each country
    for country in all_countries:
        country_deployments = [
            d for d in deployments if d["country"].title() == country
        ]
        country_code = country_deployments[0]["country_code"].lower()
        print(
            f"\n{country} ({country_code}) has {len(country_deployments)} deployments:"
        )

        total_images = 0
        for dep in sorted(country_deployments, key=lambda d: d["deployment_id"]):
            total_images += deployment_image_counter(dep, client, country_code, print_image_count)

        if print_image_count:
            print(f"Total images in {country}: {total_images}")


if __name__ == "__main__":
    with open("./credentials.json", encoding="utf-8") as config_file:
        aws_credentials = json.load(config_file)

    parser = argparse.ArgumentParser(
        description="Script for printing the deployments available on the Jasmin object store."
    )
    parser.add_argument(
        "--include_inactive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flag to include inactive deployments.",
    )
    parser.add_argument(
        "--print_image_count",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flag to print the number of images per deployment.",
    )
    parser.add_argument(
        "--subset_countries",
        nargs="+",
        default=None,
        help="Optional list to subset for specific countries (e.g. --subset_countries 'Panama' 'Thailand').",
    )
    args = parser.parse_args()

    print_deployments(
        aws_credentials,
        args.include_inactive,
        args.subset_countries,
        args.print_image_count,
    )
