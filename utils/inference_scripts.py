import torch
import pandas as pd
import os
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np
from datetime import datetime
import warnings

# ignore the pandas Future Warning
warnings.simplefilter(action="ignore", category=FutureWarning)


def classify_species(image_tensor, regional_model, regional_category_map, top_n=5):
    """
    Classify the species of the moth using the regional model.
    """

    # print('Inference for species...')
    output = regional_model(image_tensor)
    predictions = torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy()[0]

    # Sort predictions to get the indices of the top 5 scores
    top_n_indices = predictions.argsort()[-top_n:][::-1]

    # Map indices to labels and fetch their confidence scores
    index_to_label = {index: label for label, index in regional_category_map.items()}
    top_n_labels = [index_to_label[idx] for idx in top_n_indices]
    top_n_scores = [predictions[idx] for idx in top_n_indices]

    return top_n_labels, top_n_scores


def classify_order(image_tensor, order_model, order_labels, order_data_thresholds):
    """
    Classify the order of the object using the order model by Bjerge et al.
    Model and code available at: https://github.com/kimbjerge/MCC24-trap/tree/main
    """

    # print('Inference for order...')
    pred = order_model(image_tensor)
    pred = torch.nn.functional.softmax(pred, dim=1)
    predictions = pred.cpu().detach().numpy()

    predicted_label = np.argmax(predictions, axis=1)[0]
    score = predictions.max(axis=1).astype(float)[0]

    label = order_labels[predicted_label]

    return label, score


def classify_box(image_tensor, binary_model):
    """
    Classify the object as moth or non-moth using the binary model.
    """

    # print('Inference for moth/non-moth...')
    output = binary_model(image_tensor)

    predictions = torch.nn.functional.softmax(output, dim=1)
    predictions = predictions.cpu().detach().numpy()
    categories = predictions.argmax(axis=1)

    labels = {"moth": 0, "nonmoth": 1}

    index_to_label = {index: label for label, index in labels.items()}
    label = [index_to_label[cat] for cat in categories][0]
    score = predictions.max(axis=1).astype(float)[0]
    return label, score


def perform_inf(
    image_path,
    bucket_name,
    loc_model,
    binary_model,
    order_model,
    order_labels,
    regional_model,
    regional_category_map,
    proc_device,
    order_data_thresholds,
    csv_file,
    save_crops,
    box_threshold=0.995,
    top_n=5,
):
    """
    Perform inferences on an image including:
      - object detection
      - object classification
      - order classification
      - species classification
    """

    transform_loc = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_species = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    all_cols = [
        "image_path",
        "bucket_name",
        "analysis_datetime",
        "box_score",
        "box_label",
        "x_min",
        "y_min",
        "x_max",
        "y_max",  # localisation info
        "class_name",
        "class_confidence",  # binary class info
        "order_name",
        "order_confidence",  # order info
        "cropped_image_path",
    ]
    all_cols = (
        all_cols
        + ["top_" + str(i + 1) + "_species" for i in range(top_n)]
        + ["top_" + str(i + 1) + "_confidence" for i in range(top_n)]
    )

    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()
    original_width, original_height = image.size

    # print('Inference for localisation...')
    input_tensor = transform_loc(image).unsqueeze(0).to(proc_device)

    all_boxes = pd.DataFrame(columns=all_cols)

    # Perform object localisation
    with torch.no_grad():
        localisation_outputs = loc_model(input_tensor)

        # catch no crops
        if len(localisation_outputs[0]["boxes"]) == 0 or all(
            localisation_outputs[0]["scores"] < box_threshold
        ):
            df = pd.DataFrame(
                [
                    [image_path, bucket_name, str(datetime.now())]
                    + [""] * (len(all_cols) - 3),
                ],
                columns=all_cols,
            )
            if not df.empty:
                all_boxes = pd.concat([all_boxes, df])

            df.to_csv(
                f"{csv_file}",
                mode="a",
                header=not os.path.isfile(csv_file),
                index=False,
            )

        # for each detection
        for i in range(len(localisation_outputs[0]["boxes"])):
            x_min, y_min, x_max, y_max = localisation_outputs[0]["boxes"][i]
            box_score = localisation_outputs[0]["scores"].tolist()[i]
            box_label = localisation_outputs[0]["labels"].tolist()[i]

            x_min = int(int(x_min) * original_width / 300)
            y_min = int(int(y_min) * original_height / 300)
            x_max = int(int(x_max) * original_width / 300)
            y_max = int(int(y_max) * original_height / 300)

            box_width = x_max - x_min
            box_height = y_max - y_min

            if box_score < box_threshold:
                continue

            # if box height or width > half the image, skip
            # if box_width > original_width / 2 or box_height > original_height / 2:
            #    continue

            # Crop the detected region and perform classification
            cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
            cropped_tensor = (
                transform_species(cropped_image).unsqueeze(0).to(proc_device)
            )

            class_name, class_confidence = classify_box(cropped_tensor, binary_model)
            order_name, order_confidence = classify_order(
                cropped_tensor, order_model, order_labels, order_data_thresholds
            )

            # Annotate image with bounding box and class
            if class_name == "moth" or "Lepidoptera" in order_name:
                species_names, species_confidences = classify_species(
                    cropped_tensor, regional_model, regional_category_map, top_n
                )

            else:
                species_names, species_confidences = [""] * top_n, [""] * top_n

            # if save_crops then save the cropped image
            crop_path = ""
            if save_crops:
                crop_path = image_path.replace(".jpg", f"_crop{i}.jpg")
                cropped_image.save(crop_path)

            # append to csv with pandas
            df = pd.DataFrame(
                [
                    [
                        image_path,
                        bucket_name,
                        str(datetime.now()),
                        box_score,
                        box_label,
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        class_name,
                        class_confidence,
                        order_name,
                        order_confidence,
                        crop_path,
                    ]
                    + species_names
                    + species_confidences
                ],
                columns=all_cols,
            )
            if not df.empty:
                all_boxes = pd.concat([all_boxes, df])

            df.to_csv(
                f"{csv_file}",
                mode="a",
                header=not os.path.isfile(csv_file),
                index=False,
            )
