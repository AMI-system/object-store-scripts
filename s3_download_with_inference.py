#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script downloads files from an S3 bucket synchronously and performs
inference on the images. AWS credentials, S3 bucket name, and UKCEH API
credentials are loaded from a configuration file (credentials.json).
"""

import json
import boto3
import torch
import pandas as pd
import os
import argparse

from utils.aws_scripts import get_objects, get_deployments
from utils.custom_models import load_models

# Use GPU if available
print(f'  - Cuda available: {torch.cuda.is_available()}')
if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def display_menu(country, deployment, crops_interval, csv_file, rerun_existing):
    """
    Display the main menu and handle user interaction.
    """

    print("- Read in configs and credentials")

    username = aws_credentials['UKCEH_username']
    password = aws_credentials['UKCEH_password']

    all_deployments = get_deployments(username, password)

    print('- Analysing: ', country)

    country_deployments = [
        f"{d['location_name']} - {d['camera_id']}"
        for d in all_deployments
        if d["country"] == country and d["status"] == "active"
    ]

    s3_bucket_name = [
        d["country_code"]
        for d in all_deployments
        if d["country"] == country and d["status"] == "active"
    ][0].lower()

    perform_inference = True
    remove_image = True

    print('  - Removing images after analysis: ', remove_image)
    print('  - Rerun existing inferences: ', rerun_existing)
    print('  - Performing inference: ', perform_inference)

    if deployment == 'All':
        deps = country_deployments
    else:
        deps = [deployment]
    for region in deps:
        print(f'  - Deployment: {region}')
        location_name, camera_id = region.split(" - ")
        dep_id = [
            d["deployment_id"]
            for d in all_deployments
            if d["country"] == country
            and d["location_name"] == location_name
            and d["camera_id"] == camera_id
            and d["status"] == "active"
        ][0]

        prefix = f"{dep_id}/snapshot_images"
        get_objects(session,
                    aws_credentials,
                    s3_bucket_name,
                    prefix,
                    local_directory_path,
                    batch_size=100,
                    perform_inference=perform_inference,
                    remove_image=remove_image,
                    localisation_model=model_loc,
                    binary_model=classification_model,
                    order_model=order_model,
                    order_labels=order_labels,
                    species_model=regional_model,
                    species_labels=regional_category_map,
                    country=country,
                    region=region,
                    device=device,
                    order_data_thresholds=order_data_thresholds,
                    csv_file=csv_file,
                    rerun_existing=rerun_existing,
                    crops_interval=crops_interval)


if __name__ == "__main__":

    print(' - Loading models...')

    # Load AWS credentials and S3 bucket name from config file
    with open("./credentials.json", encoding="utf-8") as config_file:
        aws_credentials = json.load(config_file)

    # Initialize boto3 session
    session = boto3.Session(
        aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_credentials["AWS_REGION"],
    )

    local_directory_path = aws_credentials['directory']
    print('  - Scratch storage: ', local_directory_path)

    # date_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    # csv_file = f'{local_directory_path}/mila_outputs_{date_time}.csv'

    model_loc, classification_model, regional_model, regional_category_map, order_model, order_data_thresholds, order_labels = load_models(device)

    parser = argparse.ArgumentParser(description="Script for downloading and processing images from S3.")
    parser.add_argument("--country", type=str, help="Specify the country name")
    parser.add_argument("--deployment", type=str,
                        help="Specify the deployment name")
    parser.add_argument("--keep_crops", action=argparse.BooleanOptionalAction,
                        default=False, help="Whether to keep the crops")
    parser.add_argument("--crops_interval", type=str,
                        help="The interval for which to preserve the crops",
                        default=10)
    parser.add_argument("--csv_file", type=str,
                        help="The path to the csv file to save the results",
                        default=f'./{(parser.parse_args().country).replace(" ", "_")}_results.csv')
    parser.add_argument("--rerun_existing", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Whether to rerun images which have already been analysed")

    args = parser.parse_args()

    crops_interval = args.crops_interval
    if args.keep_crops:
        crops_interval = args.crops_interval
        print(f' - Keeping crops every {crops_interval}mins')
    else:
        print(' - Not keeping crops')
        crops_interval = None

    print(f'Saving results to: {args.csv_file}')

    # if the file doesnt exist, print headers
    csv_file = args.csv_file
    if not os.path.isfile(csv_file):
        all_boxes = pd.DataFrame(
            columns=['image_path', 'analysis_datetime',
                     'box_score', 'x_min', 'y_min', 'x_max', 'y_max',
                     'class_name', 'class_confidence',
                     'order_name', 'order_confidence',
                     'species_name', 'species_confidence',
                     'cropped_image_path']
        )
        all_boxes.to_csv(csv_file, index=False)

    display_menu(args.country, args.deployment, crops_interval, csv_file, args.rerun_existing)
