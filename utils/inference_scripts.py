import torch
import pandas as pd
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np
from datetime import datetime

# ignore the pandas Future Warning
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    country,
    region,
    device,
    order_data_thresholds,
    csv_file,
    save_crops,
):
    """
    Perform inferences on an image including:
      - object detection
      - object classification
      - order classification
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
            "bix_label",
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

    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()
    original_width, original_height = image.size

    # print('Inference for localisation...')
    input_tensor = transform_loc(image).unsqueeze(0).to(device)

    all_boxes = pd.DataFrame(
        columns=all_cols
    )

    # Perform object localization
    with torch.no_grad():
        localization_outputs = loc_model(input_tensor)

        # catch no crops
        if len(localization_outputs[0]["boxes"]) == 0:
            df = pd.DataFrame(
                [
                    [
                        image_path,
                        bucket_name,
                        str(datetime.now()),
                        'None',
                        'None',
                        '',
                        '',
                        '',
                        '',
                        '',
                        '',
                        '',
                        '',
                        '',
                    ]
                ],
                columns=all_cols,
            )
            if not df.empty:
                all_boxes = pd.concat([all_boxes, df])
            df.to_csv(
                f'{csv_file}',
                mode="a",
                header=False,
                index=False,
            )

        # for each detection
        for i in range(len(localization_outputs[0]["boxes"])):
            x_min, y_min, x_max, y_max = localization_outputs[0]["boxes"][i]
            box_score = localization_outputs[0]["scores"].tolist()[i]
            box_label = localization_outputs[0]["labels"].tolist()[i]

            x_min = int(int(x_min) * original_width / 300)
            y_min = int(int(y_min) * original_height / 300)
            x_max = int(int(x_max) * original_width / 300)
            y_max = int(int(y_max) * original_height / 300)

            box_width = x_max - x_min
            box_height = y_max - y_min

            if box_score < 0.99:
                continue

            # if box height or width > half the image, skip
            if box_width > original_width / 2 or box_height > original_height / 2:
                continue
           
            # Crop the detected region and perform classification
            cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
            cropped_tensor = transform_species(cropped_image).unsqueeze(0).to(device)

            class_name, class_confidence = classify_box(cropped_tensor, binary_model)
            order_name, order_confidence = classify_order(
                cropped_tensor, order_model, order_labels, order_data_thresholds
            )

            # if save_crops then save the cropped image
            crop_path = ""
            if order_name == "Coleoptera" or order_name == 'Heteroptera' or order_name == 'Hemiptera': 
                
                if save_crops: 
                    crop_path = image_path.split(".")[0] + f"_crop{i}.jpg"
                    cropped_image.save(crop_path)

                print(f"Potential beetle: {crop_path}")
            
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
                ],
                columns=all_cols,
            )
            if not df.empty:
                all_boxes = pd.concat([all_boxes, df])

            df.to_csv(
                f'{csv_file}',
                mode="a",
                header=False,
                index=False,
            )

