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

# Load AWS credentials and S3 bucket name from config file
with open("./credentials.json", encoding="utf-8") as config_file:
    aws_credentials = json.load(config_file)

# models = {
#     'Costa Rica': '/bask/homes/f/fspo1218/amber/projects/species_classifier/outputs/turing-costarica_v03_resnet50_2024-06-04-16-17_state.pt',
#     'Singapore': '/bask/homes/f/fspo1218/amber/projects/species_classifier/outputs/turing-singapore_v02_state_resnet50_2023-12-19-14-04.pt',
# }

# Initialize boto3 session
session = boto3.Session(
    aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"],
    region_name=aws_credentials["AWS_REGION"],
)




# class Resnet50(torch.nn.Module):
#     def __init__(self, num_classes):
#         """
#         Args:
#             config: provides parameters for model generation
#         """
#         super(Resnet50, self).__init__()
#         self.num_classes = num_classes
#         self.backbone = torchvision.models.resnet50(weights="DEFAULT")
#         out_dim = self.backbone.fc.in_features

#         self.backbone = torch.nn.Sequential(*list(self.backbone.children())[:-2])
#         self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.classifier = torch.nn.Linear(out_dim, self.num_classes, bias=False)

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)

#         return x

# class ResNet50_order(nn.Module):
#     '''ResNet-50 Architecture with pretrained weights
#     '''

#     def __init__(self, use_cbam=True, image_depth=3, num_classes=20):
#         '''Params init and build arch.
#         '''
#         super(ResNet50_order, self).__init__()

#         self.expansion = 4
#         self.out_channels = 512

#         #self.model_ft = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) # 80.86, 25.6M
#         self.model_ft = models.resnet50(pretrained=True)

#         # overwrite the 'fc' layer
#         print("In features", self.model_ft.fc.in_features)
#         self.model_ft.fc = nn.Identity() # Do nothing just pass input to output

#         # At least one layer
#         self.drop = nn.Dropout(p=0.5)
#         self.linear_lvl1 = nn.Linear(self.out_channels*self.expansion, self.out_channels)
#         self.relu_lv1 = nn.ReLU(inplace=False)
#         self.softmax_reg1 = nn.Linear(self.out_channels, num_classes)

#     def forward(self, x):
#         '''Forward propagation of pretrained ResNet-50.
#         '''
#         x = self.model_ft(x)

#         x = self.drop(x) # Dropout to add regularization

#         level_1 = self.softmax_reg1(self.relu_lv1(self.linear_lvl1(x)))
#         #level_1 = nn.Softmax(level_1)

#         return level_1

# def classify_species(image_tensor):
#     with torch.no_grad():
#         species_output = regional_model(image_tensor)
#         _, predicted_species = torch.max(species_output, 1)
#         confidence = max((torch.sigmoid(species_output)).tolist()[0])
#     species_name = list(category_map.keys())[predicted_species]
#     return species_name, confidence

# def load_models():

#     # Load the localisation model
#     weights_path = "/bask/homes/f/fspo1218/amber/data/mila_models/v1_localizmodel_2021-08-17-12-06.pt"

#     model_loc = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
#     num_classes = 2  # 1 class (object) + background
#     in_features = model_loc.roi_heads.box_predictor.cls_score.in_features
#     model_loc.roi_heads.box_predictor = (
#         torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes))
#     checkpoint = torch.load(weights_path, map_location=device)
#     state_dict = checkpoint.get("model_state_dict") or checkpoint
#     model_loc.load_state_dict(state_dict)
#     model_loc = model_loc.to(device)
#     model_loc.eval()

#     # Load the binary model
#     weights_path = "/bask/homes/f/fspo1218/amber/data/mila_models/moth-nonmoth-effv2b3_20220506_061527_30.pth"
#     labels_path = "/bask/homes/f/fspo1218/amber/data/mila_models/05-moth-nonmoth_category_map.json"
#     num_classes=2 # moth, non-moth
#     classification_model = timm.create_model("tf_efficientnetv2_b3", num_classes=num_classes, weights=None)
#     classification_model = classification_model.to(device)
#     checkpoint = torch.load(weights_path, map_location=device)
#     state_dict = checkpoint.get("model_state_dict") or checkpoint
#     classification_model.load_state_dict(state_dict)
#     classification_model.eval()

#     # Load the order model
#     savedWeights = '/bask/homes/f/fspo1218/amber/projects/MCC24-trap/model_order_060524/dhc_best_128.pth'
#     thresholdFile = '/bask/homes/f/fspo1218/amber/projects/MCC24-trap/model_order_060524/thresholdsTestTrain.csv'
#     img_size = 128
#     order_data_thresholds = pd.read_csv(thresholdFile)
#     order_labels = order_data_thresholds["ClassName"].to_list()
#     # thresholds = data_thresholds["Threshold"].to_list()
#     # means = data_thresholds["Mean"].to_list()
#     # stds = data_thresholds["Std"].to_list()
#     img_depth = 3
#     num_classes=len(order_labels)
#     model_order = ResNet50_order(num_classes=num_classes)
#     model_order.load_state_dict(torch.load(savedWeights, map_location=device))
#     model_order = model_order.to(device)
#     model_order.eval()

#     # Load the species classifier model
#     weights = '/bask/homes/f/fspo1218/amber/projects/species_classifier/outputs/turing-costarica_v03_resnet50_2024-06-04-16-17_state.pt'
#     species_category_map = json.load(open('/bask/homes/f/fspo1218/amber/data/gbif_costarica/03_costarica_data_category_map.json'))
#     num_classes = len(species_category_map)
#     species_model = Resnet50(num_classes=num_classes)
#     species_model = species_model.to(device)
#     checkpoint = torch.load(weights, map_location=device)
#     # The model state dict is nested in some checkpoints, and not in others
#     state_dict = checkpoint.get("model_state_dict") or checkpoint
#     species_model.load_state_dict(state_dict)
#     species_model.eval()

#     return model_loc, classification_model, species_model, species_category_map, model_order, order_data_thresholds, order_labels

# def classify_species(image_tensor):
#     output = regional_model(image_tensor)
#     predictions = torch.nn.functional.softmax(output, dim=1)
#     predictions = predictions.detach().numpy()
#     categories = predictions.argmax(axis=1)

#     labels = regional_category_map

#     index_to_label = {index: label for label, index in labels.items()}

#     label = [index_to_label[cat] for cat in categories][0]
#     score = 1 - predictions.max(axis=1).astype(float)[0]
#     return label, score

# def classify_order(image_tensor):
#     augment=False
#     visualize=False
#     #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
#     pred = order_model(image_tensor)


#     predictions = pred.cpu().detach().numpy()
#     predicted_label = np.argmax(predictions, axis=1)[0]
#     print('preds:', predictions)
#     print('pred labels:', predicted_label)

#     label = order_labels[predicted_label]
#     confidence_value = norm.cdf(predictions[0][predicted_label],
#                                 order_data_thresholds['Mean'][predicted_label],
#                                 order_data_thresholds['Std'][predicted_label])
#     confidence_value = round(confidence_value*10000)/100

#     return label, confidence_value

# def classify_box(image_tensor):
#     output = classification_model(image_tensor)

#     predictions = torch.nn.functional.softmax(output, dim=1)

#     predictions = predictions.detach().numpy()

#     categories = predictions.argmax(axis=1)

#     labels = {'moth': 0, 'nonmoth': 1}

#     index_to_label = {index: label for label, index in labels.items()}

#     label = [index_to_label[cat] for cat in categories][0]
#     score = predictions.max(axis=1).astype(float)[0]
#     return label, score


# def perform_inf(image_path, loc_model, binary_model, order_model, regional_model, country, region):
#     """Perform inference on an image."""
#     image = Image.open(image_path).convert('RGB')
#     original_image = image.copy()
#     original_width, original_height = image.size
#     input_tensor = transform(image).unsqueeze(0).to(device)

#     all_boxes = pd.DataFrame(columns=['image_path',
#                                   'box_score', 'x_min', 'y_min', 'x_max', 'y_max', #localisation info
#                                   'class_name', 'class_confidence', # binary class info
#                                   'order_name', 'order_confidence', # order info
#                                   'species_name', 'species_confidence']) # species info

#       # Perform object localization
#     with torch.no_grad():
#         localization_outputs = model_loc(input_tensor)

#         print(image_path)
#         print('Number of objects:', len(localization_outputs[0]['boxes']))

#         # for each detection
#         for i in range(len(localization_outputs[0]['boxes'])):
#             x_min, y_min, x_max, y_max = localization_outputs[0]['boxes'][i]
#             box_score = localization_outputs[0]['scores'].tolist()[i]

#             x_min = int(int(x_min) * original_width / 300)
#             y_min = int(int(y_min) * original_height / 300)
#             x_max = int(int(x_max) * original_width / 300)
#             y_max = int(int(y_max) * original_height / 300)

#             box_width = x_max - x_min
#             box_height = y_max - y_min

#             # if box heigh or width > half the image, skip
#             if box_width > original_width / 2 or box_height > original_height / 2:
#                 continue

#             # if confidence below threshold
#             if box_score <= 0.1:
#                 continue

#             # Crop the detected region and perform classification
#             cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
#             cropped_tensor = transform_species(cropped_image).unsqueeze(0)

#             class_name, class_confidence = classify_box(cropped_tensor)
#             order_name, order_confidence = classify_order(cropped_tensor)


#             # Annotate image with bounding box and class
#             if class_name == 'moth':
#                 # Perform the species classification
#                 print('...Performing the inference')
#                 species_name, species_confidence = classify_species(cropped_tensor)

#                 draw = ImageDraw.Draw(original_image)
#                 draw.rectangle([x_min, y_min, x_max, y_max], outline='green', width=3)
#                 draw.text((x_min, y_min - 10), species_name + " , %.3f " % species_confidence, fill='green')

#             else:
#                 species_name, species_confidence = None, None
#                 draw = ImageDraw.Draw(original_image)
#                 draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
#                 draw.text((x_min, y_min - 10), f'order: {order_name}, binary: {class_name}', fill='red')

#             draw.text((x_min, y_max), str(box_score), fill='black')

#             # append to csv with pandas
#             df = pd.DataFrame([[image_path,
#                                 box_score, x_min, y_min, x_max, y_max,
#                                 class_name, class_confidence ,
#                                 order_name, order_confidence,
#                                 species_name, species_confidence]],
#                               columns=['image_path',
#                                       'box_score', 'x_min', 'y_min', 'x_max', 'y_max',
#                                       'class_name', 'class_confidence',
#                                       'order_name', 'order_confidence',
#                                       'species_name', 'species_confidence'])
#             all_boxes = pd.concat([all_boxes, df])
#             df.to_csv(csv_file, mode='a', header=False, index=False)


#         if (all_boxes['class_name'] == 'moth').any():
#             print('...Moth Detected')
#             original_image.save(os.path.basename(image_path))


# def get_deployments(username, password):
#     """Fetch deployments from the API with authentication."""
#     try:
#         url = "https://connect-apps.ceh.ac.uk/ami-data-upload/get-deployments/"
#         response = requests.get(
#             url, auth=HTTPBasicAuth(username, password), timeout=600
#         )

#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.HTTPError as err:
#         print(f"HTTP Error: {err}")
#         if response.status_code == 401:
#             print("Wrong username or password. Try again!")
#         sys.exit(1)
#     except Exception as err:
#         print(f"Error: {err}")
#         sys.exit(1)


# def download_object(s3_client, bucket_name, key, download_path, perform_inference=False,
#                     remove_image=False, localisation_model=None, binary_model=None, order_model=None, species_model=None, country='UK', region='UKCEH'):
#     """
#     Download a single object from S3 synchronously.
#     """
#     try:
#         s3_client.download_file(
#             bucket_name, key, download_path, Config=transfer_config
#         )
#         if perform_inference:
#             perform_inf(download_path, loc_model = localisation_model, binary_model=binary_model,
#                         order_model=order_model, regional_model=species_model, country=country, region=region)
#         if remove_image:
#             os.remove(download_path)
#     except Exception as e:
#         print(f"Error downloading {bucket_name}/{key}: {e}")


# def download_batch(s3_client, bucket_name, keys, local_path, perform_inference=False,
#                    remove_image=False, localisation_model=None, binary_model=None, order_model=None, species_model=None, country='UK', region='UKCEH'):
#     """
#     Download a batch of objects from S3.
#     """
#     for key in keys:
#         file_path, filename = os.path.split(key)
#         os.makedirs(os.path.join(local_path, file_path), exist_ok=True)
#         download_path = os.path.join(local_path, file_path, filename)
#         download_object(s3_client, bucket_name, key, download_path,
#                         perform_inference, remove_image, localisation_model, binary_model, order_model, species_model, country, region)


# def count_files(s3_client, bucket_name, prefix):
#     """
#     Count number of files for a given prefix.
#     """
#     paginator = s3_client.get_paginator("list_objects_v2")
#     operation_parameters = {"Bucket": bucket_name, "Prefix": prefix}
#     page_iterator = paginator.paginate(**operation_parameters)

#     count = 0
#     for page in page_iterator:
#         count += page.get("KeyCount", 0)

#     return count


# def get_objects(bucket_name, key,
#                 local_path,
#                 batch_size=100,
#                 perform_inference=False,
#                 remove_image=False,
#                 localisation_model=None,
#                 binary_model=None,
#                 order_model=None,
#                 species_model=None,
#                country='UK', region='UKCEH'):
#     """
#     Fetch objects from the S3 bucket and download them synchronously in batches.
#     """
#     s3_client = session.client("s3", endpoint_url=aws_credentials["AWS_URL_ENDPOINT"])

#     total_files = count_files(s3_client, bucket_name, key)

#     paginator = s3_client.get_paginator("list_objects_v2")
#     operation_parameters = {"Bucket": bucket_name, "Prefix": key}
#     page_iterator = paginator.paginate(**operation_parameters)

#     progress_bar = tqdm.tqdm(total=total_files, desc="Download files from server synchronously")

#     keys = []
#     for page in page_iterator:
#         for obj in page.get("Contents", []):
#             keys.append(obj["Key"])

#             if len(keys) >= batch_size:
#                 download_batch(s3_client, bucket_name, keys, local_path, perform_inference,
#                                remove_image, localisation_model, binary_model, order_model, species_model, country, region)
#                 keys = []
#                 progress_bar.update(batch_size)
#         if keys:
#             download_batch(s3_client, bucket_name, keys, local_path, perform_inference,
#                            remove_image, localisation_model, binary_model, order_model, species_model, country, region)
#             progress_bar.update(len(keys))

#     progress_bar.close()


# def clear_screen():
#     """Clear the terminal screen."""
#     os.system("cls" if os.name == "nt" else "clear")


# def get_input(prompt):
#     """Get user input."""
#     return input(prompt + ": ")


# def get_choice(prompt, options):
#     """Get user's choice from a list of options."""
#     print(prompt)
#     for i, option in enumerate(options, start=1):
#         print(f"{i}. {option}")
#     while True:
#         choice = input("Choose an option (enter the number): ")
#         if choice.isdigit() and 1 <= int(choice) <= len(options):
#             return options[int(choice) - 1]
#         print("Invalid choice. Please try again.")


def display_menu(country, deployment):
    """Display the main menu and handle user interaction."""
    # clear_screen()
    print("Download Files")
    print("============\n")

    username = aws_credentials['UKCEH_username']
    password = aws_credentials['UKCEH_password']

    all_deployments = get_deployments(username, password)

    countries = list({d["country"] for d in all_deployments if d["status"] == "active"})
    #country = get_choice("Countries:", countries)
    #country = 'Costa Rica'
    print('Analysing: ', country)


    country_deployments = [
        f"{d['location_name']} - {d['camera_id']}"
        for d in all_deployments
        if d["country"] == country and d["status"] == "active"
    ]
    country_deployments = country_deployments

    print('Deployments: ',  deployment)


    data_types = ["snapshot_images", "audible_recordings", "ultrasound_recordings"]
    data_type = "snapshot_images"

    s3_bucket_name = [
        d["country_code"]
        for d in all_deployments
        if d["country"] == country and d["status"] == "active"
    ][0].lower()

    local_directory_path = aws_credentials['directory']
    print('Using ', local_directory_path, ' as scratch storage')

    perform_inference = True
    remove_image = True

    print('Removing images', remove_image)
    print('Performing inference', perform_inference)

    # regional_model = models[country]
    # print(f"Loading model for {country}: {regional_model}")

    #---------
    if deployment == 'All':
        deps = country_deployments
    else :
        deps = [deployment]
    for region in deps:
        print(region)
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
        print(prefix)
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
    print('Loading models...')
    date_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    csv_file = f'/bask/homes/f/fspo1218/amber/projects/object-store-scripts/data/mila_outputs_{date_time}.csv'

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
    args = parser.parse_args()

    display_menu(args.country, args.deployment)
