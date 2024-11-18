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


def download_and_inference(
    country,
    deployment,
    crops_interval,
    rerun_existing,
    local_directory_path,
    perform_inference,
    remove_image,
):
    """
    Display the main menu and handle user interaction.
    """

    output_cols = [
        "image_path",
        "bucket_name",
        "analysis_datetime",
        "box_score",
        "box_label",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "class_name",
        "class_confidence",
        "order_name",
        "order_confidence",
        "cropped_image_path",
    ]

    username = aws_credentials["UKCEH_username"]
    password = aws_credentials["UKCEH_password"]

    print(f"\033[93m - Removing images after analysis: {remove_image}\033[0m")
    print(f"\033[93m - Performing inference: {perform_inference}\033[0m")
    print(f"\033[93m - Rerun existing inferences: {rerun_existing}\033[0m")

    all_deployments = get_deployments(username, password)

    print(f"\033[96m\033[1mAnalysing: {country}\033[0m\033[0m")

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

    if deployment == "All":
        deps = country_deployments
    else:
        deps = [deployment]

    # loop through each deployment
    for region in deps:
        location_name, camera_id = region.split(" - ")
        dep_id = [
            d["deployment_id"]
            for d in all_deployments
            if d["country"] == country
            and d["location_name"] == location_name
            and d["camera_id"] == camera_id
            and d["status"] == "active"
        ][0]
        print(f"\033[96m - Deployment: {region}, id: {dep_id} \033[0m")

        # if the file doesnt exist, print headers
        csv_file = os.path.abspath(
            f"{local_directory_path}/{dep_id}/{dep_id}_results.csv"
        )
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        print(f"\033[93m - Saving results to: {csv_file}\033[0m")

        if not os.path.isfile(csv_file):
            all_boxes = pd.DataFrame(columns=output_cols)
            all_boxes.to_csv(csv_file, index=False)

        prefix = f"{dep_id}/snapshot_images"
        get_objects(
            session,
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
            country=country,
            region=region,
            device=device,
            order_data_thresholds=order_data_thresholds,
            csv_file=csv_file,
            rerun_existing=rerun_existing,
            crops_interval=crops_interval,
        )
        print("\N{White Heavy Check Mark}\033[0m\033[0m")


if __name__ == "__main__":

    # Use GPU if available

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            "\033[95m\033[1mCuda available, using GPU "
            + "\N{White Heavy Check Mark}\033[0m\033[0m"
        )
    else:
        device = torch.device("cpu")
        print(
            "\033[95m\033[1mCuda not available, using CPU "
            + "\N{Cross Mark}\033[0m\033[0m"
        )

    print("\033[96m\033[1mInitialising the JASMINE session...\033[0m\033[0m", end="")
    # Load AWS credentials and S3 bucket name from config file
    with open("./credentials.json", encoding="utf-8") as config_file:
        aws_credentials = json.load(config_file)
    session = boto3.Session(
        aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
        region_name=aws_credentials["AWS_REGION"],
    )
    print("\N{White Heavy Check Mark}")

    parser = argparse.ArgumentParser(
        description="Script for downloading and processing images from S3."
    )
    parser.add_argument("--country", type=str, help="Specify the country name")
    parser.add_argument("--deployment", type=str, help="Specify the deployment name")
    parser.add_argument(
        "--keep_crops",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to keep the crops",
    )
    parser.add_argument(
        "--perform_inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to perform the inference",
    )
    parser.add_argument(
        "--remove_image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to remove the raw image after inference",
    )
    parser.add_argument(
        "--rerun_existing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to rerun images which have already been analysed",
    )
    parser.add_argument(
        "--data_storage_path",
        type=str,
        help="The path to scratch data storage",
        default="./data/",
    )
    parser.add_argument(
        "--binary_model_path",
        type=str,
        help="The path to the binary model weights",
        default="./models/moth-nonmoth-effv2b3_20220506_061527_30.pth",
    )
    parser.add_argument(
        "--order_model_path",
        type=str,
        help="The path to the binary model weights",
        default="./models/dhc_best_128.pth",
    )
    parser.add_argument(
        "--order_threshold_path",
        type=str,
        help="The path to the binary model weights",
        default="./models/thresholdsTestTrain.csv",
    )
    parser.add_argument(
        "--localisation_model_path",
        type=str,
        help="The path to the binary model weights",
        default="./models/v1_localizmodel_2021-08-17-12-06.pt",
    )

    args = parser.parse_args()

    print("\033[96m\033[1mLoading in models...\033[0m\033[0m", end="")
    (
        model_loc,
        classification_model,
        order_model,
        order_data_thresholds,
        order_labels,
    ) = load_models(device, args.localisation_model_path, args.binary_model_path, args.order_model_path, args.order_threshold_path)
    print("\N{White Heavy Check Mark}")    

    # check that the data storage path exists
    data_storage_path = os.path.abspath(args.data_storage_path)
    if not os.path.isdir(data_storage_path):
        os.makedirs(data_storage_path)

    print("\033[93m\033[1m" + "Pipeline parameters" + "\033[0m\033[0m")
    print(f"\033[93m - Scratch and crops storage: {data_storage_path}\033[0m")

    if args.keep_crops:
        crops_interval = args.crops_interval
        print(f"\033[93m - Keeping crops every {crops_interval}mins\033[0m")
    else:
        print("\033[93m - Not keeping crops\033[0m")
        crops_interval = None

    download_and_inference(
        args.country,
        args.deployment,
        int(crops_interval),
        args.rerun_existing,
        data_storage_path,
        args.perform_inference,
        args.remove_image,
    )
