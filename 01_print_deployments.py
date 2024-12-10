#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script downloads files from an S3 bucket synchronously and performs
inference on the images. AWS credentials, S3 bucket name, and UKCEH API
credentials are loaded from a configuration file (credentials.json).
"""

import json
import boto3
import os
import argparse

import requests
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


def print_deployments(
    include_inactive=False, subset_countries=None, print_image_count=True
):
    """
    Provide the deployments available through the object store.
    """
    import sys

    print(sys.version)

    # Get all deployments
    username = aws_credentials["UKCEH_username"]
    password = aws_credentials["UKCEH_password"]
    all_deployments = get_deployments(username, password)

    session = boto3.Session(
        aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_credentials["AWS_REGION"],
    )
    s3_client = session.client("s3", endpoint_url=aws_credentials["AWS_URL_ENDPOINT"])

    if not include_inactive:
        act_string = " active "
        all_deployments = [x for x in all_deployments if x["status"] == "active"]
    else:
        act_string = " "

    # Loop through each country to print deployment information
    all_countries = list(set([dep["country"].title() for dep in all_deployments]))

    if subset_countries is not None:
        subset_countries = [x.title() for x in subset_countries]
        not_included_countries = [x for x in subset_countries if x not in all_countries]
        for missing in not_included_countries:
            print(
                f"\033[1mWARNING: {missing} does not have any {act_string}deployments, check spelling\033[0m"
            )
        all_countries = [x for x in all_countries if x in subset_countries]

    for country in all_countries:
        country_depl = [x for x in all_deployments if x["country"] == country]
        country_code = list(set([x["country_code"] for x in country_depl]))[0]
        print(
            "\n\033[1m"
            + country
            + " ("
            + country_code
            + ") has "
            + str(len(country_depl))
            + act_string
            + "deployments:\033[0m"
        )
        all_deps = list(set([x["deployment_id"] for x in country_depl]))

        total_images = 0
        for dep in sorted(all_deps):
            dep_info = [x for x in country_depl if x["deployment_id"] == dep][0]
            print(
                f"\033[1m - Deployment ID: {dep_info['deployment_id']}, Name: {dep_info['location_name']}, Deployment Key: '{dep_info['location_name']} - {dep_info['camera_id']}'\033[0m"
            )
            print(
                f"   Location ID: {dep_info['location_id']}, Country code: {dep_info['country_code'].lower()}, Latitute: {dep_info['lat']}, Longitute: {dep_info['lon']}, Camera ID: {dep_info['camera_id']}, System ID: {dep_info['system_id']}, Status: {dep_info['status']}"
            )

            # get the number of images for this deployment
            prefix = f"{dep_info['deployment_id']}/snapshot_images"
            bucket_name = dep_info["country_code"].lower()

            if print_image_count:
                count = count_files(s3_client, bucket_name, prefix)
                total_images = total_images + count
                print(f" - This deployment has \033[1m{count}\033[0m images.\n")

        if print_image_count:
            print(
                f"\033[1m - {country} has {total_images}\033[0m images total\033[0m\n"
            )


if __name__ == "__main__":
    with open("./credentials.json", encoding="utf-8") as config_file:
        aws_credentials = json.load(config_file)
    session = boto3.Session(
        aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_credentials["AWS_REGION"],
    )

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
        help="Optional list to subset for specific countries (e.g. --subset_countries 'Panama' 'Thailand').",
        default=None,
    )
    args = parser.parse_args()

    print_deployments(
        args.include_inactive, args.subset_countries, args.print_image_count
    )
