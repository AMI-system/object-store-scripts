#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script downloads files from an S3 bucket synchronously.
AWS credentials and S3 bucket name are loaded from a configuration file
(credentials.json).
"""

import sys
import os
import getpass
import json
import requests
from requests.auth import HTTPBasicAuth
import boto3
from boto3.s3.transfer import TransferConfig
import tqdm
import csv
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import pandas as pd
import timm
import datetime
import argparse

device = torch.device('cpu')

# Load AWS credentials and S3 bucket name from config file
with open("./credentials.json", encoding="utf-8") as config_file:
    aws_credentials = json.load(config_file)

models = {
    'Costa Rica': '/bask/homes/f/fspo1218/amber/projects/species_classifier/outputs/turing-costarica_v03_resnet50_2024-06-04-16-17_state.pt',
    'Singapore': '/bask/homes/f/fspo1218/amber/projects/species_classifier/outputs/turing-singapore_v02_state_resnet50_2023-12-19-14-04.pt',
}

# Initialize boto3 session
session = boto3.Session(
    aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
    region_name=aws_credentials["AWS_REGION"],
)

# Configure the transfer to optimize the download
transfer_config = TransferConfig(
    max_concurrency=20,  # Increase the number of concurrent transfers
    multipart_threshold=8 * 1024 * 1024,  # 8MB
    max_io_queue=1000,
    io_chunksize=262144,  # 256KB
)

# csv_file = '/bask/homes/f/fspo1218/amber/projects/object-store-scripts/mila_outputs.csv'

transform = transforms.Compose([
    transforms.Resize((300, 300)),  # Assuming models require 300x300 input images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_species = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

class Resnet50(torch.nn.Module):
    def __init__(self, num_classes):
        """
        Args:
            config: provides parameters for model generation
        """
        super(Resnet50, self).__init__()
        self.num_classes = num_classes
        self.backbone = torchvision.models.resnet50(weights="DEFAULT")
        out_dim = self.backbone.fc.in_features

        self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-2])
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = torch.nn.Linear(out_dim, self.num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

def classify_species(image_tensor):
    with torch.no_grad():
        species_output = regional_model(image_tensor)
        _, predicted_species = torch.max(species_output, 1)
        confidence = max((torch.sigmoid(species_output)).tolist()[0])
    species_name = list(category_map.keys())[predicted_species]
    return species_name, confidence

def load_models():
    weights_path = "/bask/homes/f/fspo1218/amber/data/mila_models/v1_localizmodel_2021-08-17-12-06.pt"

    model_loc = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # 1 class (object) + background
    in_features = model_loc.roi_heads.box_predictor.cls_score.in_features
    model_loc.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    model_loc.load_state_dict(state_dict)
    model_loc = model_loc.to(device)
    model_loc.eval()

    weights_path = "/bask/homes/f/fspo1218/amber/data/mila_models/moth-nonmoth-effv2b3_20220506_061527_30.pth"
    labels_path = "/bask/homes/f/fspo1218/amber/data/mila_models/05-moth-nonmoth_category_map.json"

    num_classes = 2
    classification_model = timm.create_model(
        "tf_efficientnetv2_b3",
        num_classes=num_classes,
        weights=None,
    )
    classification_model = classification_model.to(device)
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    classification_model.load_state_dict(state_dict)
    classification_model.eval()

    weights = '/bask/homes/f/fspo1218/amber/projects/species_classifier/outputs/turing-costarica_v03_resnet50_2024-06-04-16-17_state.pt'
    category_map = json.load(open('/bask/homes/f/fspo1218/amber/data/gbif_costarica/03_costarica_data_category_map.json'))

    num_classes = len(category_map)
    species_model = Resnet50(num_classes=num_classes)
    species_model = species_model.to(device)
    checkpoint = torch.load(weights, map_location=device)
    # The model state dict is nested in some checkpoints, and not in others
    state_dict = checkpoint.get("model_state_dict") or checkpoint

    species_model.load_state_dict(state_dict)
    species_model.eval()

    return model_loc, classification_model, species_model, category_map

def perform_inf(image_path, loc_model, binary_model, regional_model, country, region):
    """Perform inference on an image."""
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    original_width, original_height = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    all_boxes = pd.DataFrame(columns=['image_path', 'timestamp', 'country', 'deployment', 'class', 'class_confidence', 'x_min', 'y_min', 'x_max', 'y_max', 'species_name', 'species_confidence'])

    # Perform object localization
    with torch.no_grad():
        localization_outputs = model_loc(input_tensor)

        # for each detection
        for i in range(len(localization_outputs[0]['boxes'])):
            x_min, y_min, x_max, y_max = localization_outputs[0]['boxes'][i]
            score = localization_outputs[0]['scores'][i]

            x_min = int(int(x_min) * original_width / 300)
            y_min = int(int(y_min) * original_height / 300)
            x_max = int(int(x_max) * original_width / 300)
            y_max = int(int(y_max) * original_height / 300)

            box_width = x_max - x_min
            box_height = y_max - y_min

            # if box height or width > half the image, skip
            if box_width > original_width / 2 or box_height > original_height / 2:
                continue

            # Crop the detected region and perform classification
            cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
            cropped_tensor = transform_species(cropped_image).unsqueeze(0)
            # cropped_species_tensor = transform_species(cropped_image).unsqueeze(0)

            with torch.no_grad():
                classification_output = classification_model(cropped_tensor)
                _, predicted_class = torch.max(classification_output, 1)
                confidence = max((torch.sigmoid(classification_output)).tolist()[0])

            class_name = 'non-moth' if predicted_class.item() == 1 else 'moth'

            # Annotate image with bounding box and class
            if class_name == 'moth':
                # Perform the species classification
                species_name, species_confidence = classify_species(cropped_tensor)

                draw = ImageDraw.Draw(original_image)
                draw.rectangle([x_min, y_min, x_max, y_max], outline='green', width=3)
                draw.text((x_min, y_min - 10), species_name + " , %.3f " % species_confidence, fill='green')

            else:
                species_name, species_confidence = None, None

            # append to csv with pandas
            df = pd.DataFrame([[image_path, datetime.datetime.now(), country, region, class_name, confidence, x_min, y_min, x_max, y_max, species_name, species_confidence]],
                              columns=['image_path', 'timestamp', 'country', 'deployment', 'class', 'class_confidence', 'x_min', 'y_min', 'x_max', 'y_max', 'species_name', 'species_confidence'])
            all_boxes = pd.concat([all_boxes, df], ignore_index=True)

    return original_image, all_boxes

def upload_file(s3_bucket, local_file_path, s3_object_name=None):
    if s3_object_name is None:
        s3_object_name = os.path.basename(local_file_path)

    s3_client = session.client('s3')
    s3_client.upload_file(local_file_path, s3_bucket, s3_object_name, Config=transfer_config)

def save_image_with_boxes(image_with_boxes, output_image_path):
    image_with_boxes.save(output_image_path)

def main(country, region, local_directory_path):
    s3_bucket = aws_credentials["S3_BUCKET_NAME"]

    # Load the models
    model_loc, classification_model, regional_model, category_map = load_models()

    # List all the image files in the local directory
    image_files = [f for f in os.listdir(local_directory_path) if os.path.isfile(os.path.join(local_directory_path, f))]

    for image_file in tqdm.tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(local_directory_path, image_file)

        # Perform inference on the image
        image_with_boxes, df = perform_inf(image_path, model_loc, classification_model, regional_model, country, region)

        # Save the output image with bounding boxes locally
        output_image_path = os.path.join(local_directory_path, f"boxed_{image_file}")
        save_image_with_boxes(image_with_boxes, output_image_path)

        # Save the dataframe as a CSV file
        csv_file_path = os.path.join(local_directory_path, "mila_outputs.csv")
        if os.path.exists(csv_file_path):
            df.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file_path, mode='w', header=True, index=False)

        # Upload the CSV file to S3
        upload_file(s3_bucket, csv_file_path, s3_object_name="mila_outputs.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and upload results to S3.")
    parser.add_argument("country", type=str, help="Country of the deployment.")
    parser.add_argument("region", type=str, help="Region of the deployment.")
    parser.add_argument("local_directory_path", type=str, help="Local directory path containing images.")
    args = parser.parse_args()

    main(args.country, args.region, args.local_directory_path)
