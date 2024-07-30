import torch
import torchvision
import torch.nn as nn
import timm
from torchvision import models
import pandas as pd
import json
from scipy.stats import norm
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np

from utils.custom_models import Resnet50_species, ResNet50_order, load_models


def classify_species(image_tensor, regional_model, regional_category_map):
    # print('Inference for species...')
    output = regional_model(image_tensor)
    predictions = torch.nn.functional.softmax(output, dim=1)
    predictions = predictions.detach().numpy()
    categories = predictions.argmax(axis=1)

    labels = regional_category_map

    index_to_label = {index: label for label, index in labels.items()}

    label = [index_to_label[cat] for cat in categories][0]
    score = 1 - predictions.max(axis=1).astype(float)[0]
    return label, score

def classify_order(image_tensor, order_model, order_labels, order_data_thresholds):
    # print('Inference for order...')
    augment=False
    visualize=False
    #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
    pred = order_model(image_tensor)

    pred = torch.nn.functional.softmax(pred, dim=1) #.cpu().numpy()[0]) * 100
    
    
    predictions = pred.cpu().detach().numpy()
    predicted_label = np.argmax(predictions, axis=1)[0]
    score = predictions.max(axis=1).astype(float)[0]
    
    label = order_labels[predicted_label]
    # confidence_value = norm.cdf(predictions[0][predicted_label],
    #                             order_data_thresholds['Mean'][predicted_label],
    #                             order_data_thresholds['Std'][predicted_label])
    # confidence_value = round(confidence_value*10000)/100


    return label, score # confidence_value

def classify_box(image_tensor, binary_model):
    # print('Inference for moth/non-moth...')
    output = binary_model(image_tensor)

    predictions = torch.nn.functional.softmax(output, dim=1)

    predictions = predictions.detach().numpy()

    categories = predictions.argmax(axis=1)

    labels = {'moth': 0, 'nonmoth': 1}

    index_to_label = {index: label for label, index in labels.items()}

    label = [index_to_label[cat] for cat in categories][0]
    score = predictions.max(axis=1).astype(float)[0]
    return label, score


def perform_inf(image_path, loc_model, binary_model, order_model, order_labels,
                regional_model, regional_category_map, country, region, device,
                order_data_thresholds, csv_file, save_crops):
    """Perform inference on an image."""
    
    transform_loc = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_species = transforms.Compose(
            [
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5, 0.5], std=[0.5,0.5, 0.5]),
            ]
        )

    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    original_width, original_height = image.size
    # print('Inference for localisation...')
    input_tensor = transform_loc(image).unsqueeze(0).to(device)

    all_boxes = pd.DataFrame(columns=['image_path',
                                  'box_score', 'x_min', 'y_min', 'x_max', 'y_max', #localisation info
                                  'class_name', 'class_confidence', # binary class info
                                  'order_name', 'order_confidence', # order info
                                  'species_name', 'species_confidence']) # species info

      # Perform object localization
    with torch.no_grad():
        localization_outputs = loc_model(input_tensor)

        # print(image_path)
        # print('Number of objects:', len(localization_outputs[0]['boxes']))

        # for each detection
        for i in range(len(localization_outputs[0]['boxes'])):
            x_min, y_min, x_max, y_max = localization_outputs[0]['boxes'][i]
            box_score = localization_outputs[0]['scores'].tolist()[i]

            x_min = int(int(x_min) * original_width / 300)
            y_min = int(int(y_min) * original_height / 300)
            x_max = int(int(x_max) * original_width / 300)
            y_max = int(int(y_max) * original_height / 300)

            box_width = x_max - x_min
            box_height = y_max - y_min

            # if box heigh or width > half the image, skip
            if box_width > original_width / 2 or box_height > original_height / 2:
                continue

            # if confidence below threshold
            if box_score <= 0.1:
                continue

            # Crop the detected region and perform classification
            cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
            cropped_tensor = transform_species(cropped_image).unsqueeze(0)

            class_name, class_confidence = classify_box(cropped_tensor, binary_model)
            order_name, order_confidence = classify_order(cropped_tensor, order_model, order_labels, order_data_thresholds)


            # Annotate image with bounding box and class
            if class_name == 'moth':
                # Perform the species classification
                # print('...Performing the inference')
                species_name, species_confidence = classify_species(cropped_tensor, regional_model, regional_category_map)

                draw = ImageDraw.Draw(original_image)
                draw.rectangle([x_min, y_min, x_max, y_max], outline='green', width=3)
                draw.text((x_min, y_min - 10), species_name + " , %.3f " % species_confidence, fill='green')

            else:
                species_name, species_confidence = None, None
                draw = ImageDraw.Draw(original_image)
                draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
                draw.text((x_min, y_min - 10), f'order: {order_name}, binary: {class_name}', fill='red')

            draw.text((x_min, y_max), str(box_score), fill='black')

            # append to csv with pandas
            df = pd.DataFrame([[image_path,
                                box_score, x_min, y_min, x_max, y_max,
                                class_name, class_confidence ,
                                order_name, order_confidence,
                                species_name, species_confidence]],
                              columns=['image_path',
                                      'box_score', 'x_min', 'y_min', 'x_max', 'y_max',
                                      'class_name', 'class_confidence',
                                      'order_name', 'order_confidence',
                                      'species_name', 'species_confidence'])
            all_boxes = pd.concat([all_boxes, df])
            df.to_csv(csv_file, mode='a', header=False, index=False)


        # if (all_boxes['class_name'] == 'moth').any():
        #     print('...Moth Detected')
        #     original_image.save(os.path.basename(image_path))