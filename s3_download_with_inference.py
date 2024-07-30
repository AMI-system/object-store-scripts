#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script downloads files from an S3 bucket synchronously and performs inference on the images.
AWS credentials, S3 bucket name, and UKCEH API credentials are loaded from a configuration file
(credentials.json).
"""

import sys
import os
import getpass
import json
import requests
from requests.auth import HTTPBasicAuth
import boto3
import torch
import tqdm
import csv
import pandas as pd

import datetime

import argparse
import numpy as np

from utils.aws_scripts import get_objects, get_deployments
from utils.custom_models import Resnet50_species, ResNet50_order, load_models

device = torch.device('cpu')



def display_menu(country, deployment):
    """Display the main menu and handle user interaction."""

    print("- Read in configs and credentials")


    username = aws_credentials['UKCEH_username']
    password = aws_credentials['UKCEH_password']

    all_deployments = get_deployments(username, password)

    countries = list({d["country"] for d in all_deployments if d["status"] == "active"})
    print('- Analysing: ', country)

    country_deployments = [
        f"{d['location_name']} - {d['camera_id']}"
        for d in all_deployments
        if d["country"] == country and d["status"] == "active"
    ]
    country_deployments = country_deployments


    data_types = ["snapshot_images", "audible_recordings", "ultrasound_recordings"]
    data_type = "snapshot_images"

    s3_bucket_name = [
        d["country_code"]
        for d in all_deployments
        if d["country"] == country and d["status"] == "active"
    ][0].lower()

    

    perform_inference = True
    remove_image = True

    print('  - Removing images after analysis: ', remove_image)
    print('  - Performing inference: ', perform_inference)

    #---------
    if deployment == 'All':
        deps = country_deployments
    else :
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



        prefix = f"{dep_id}/{data_type}"
        get_objects(session, aws_credentials,
                    s3_bucket_name, prefix, local_directory_path,
                    batch_size=100,
                    perform_inference=perform_inference,
                    remove_image=remove_image,
                    localisation_model=model_loc,
                    binary_model=classification_model,
                    order_model=order_model,
                    order_labels=order_labels,
                    species_model=regional_model,
                    species_labels=regional_category_map,
                   country=country, region=region, device=device,
                   order_data_thresholds=order_data_thresholds, 
                   csv_file=csv_file)

if __name__ == "__main__":
    print('  - Loading models...')
    
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
    
    date_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    csv_file = f'{local_directory_path}/mila_outputs_{date_time}.csv'

    all_boxes = pd.DataFrame(
        columns=['image_path', 'box_score', 'x_min', 'y_min', 'x_max', 'y_max',
                 'class_name', 'class_confidence', 'order_name', 'order_confidence',
                 'species_name', 'species_confidence']
    )
    all_boxes.to_csv(csv_file, index=False)


    model_loc, classification_model, regional_model, regional_category_map, order_model, order_data_thresholds, order_labels = load_models(device)

    parser = argparse.ArgumentParser(description="Script for downloading and processing images from S3.")
    parser.add_argument("--country", type=str, help="Specify the country name")
    parser.add_argument("--deployment", type=str, help="Specify the deployment name")
    parser.add_argument("--keep_crops", action=argparse.BooleanOptionalAction, help="Whether to keep the crops")
    parser.add_argument("--crops_interval", type=str, help="The interval for which to preserve the crops")

    args = parser.parse_args()

    display_menu(args.country, args.deployment)
