# utils/custom_models.py

import torch
import torchvision
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import timm
from torchvision import models
import pandas as pd
import json


class Resnet50_species(torch.nn.Module):
    def __init__(self, num_classes):
        """
        Args:
            config: provides parameters for model generation
        """
        super(Resnet50_species, self).__init__()
        self.num_classes = num_classes
        self.backbone = torchvision.models.resnet50(weights="DEFAULT")
        out_dim = self.backbone.fc.in_features

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(out_dim, self.num_classes, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class ResNet50_order(nn.Module):
    """ResNet-50 Architecture with pretrained weights"""

    def __init__(self, use_cbam=True, image_depth=3, num_classes=20):
        """Params init and build arch."""
        super(ResNet50_order, self).__init__()

        self.expansion = 4
        self.out_channels = 512

        self.model_ft = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # pretrained=True) # 80.86, 25.6M
        # self.model_ft = models.resnet50(pretrained=True)

        # overwrite the 'fc' layer
        # print("In features", self.model_ft.fc.in_features)
        self.model_ft.fc = nn.Identity()  # Do nothing just pass input to output

        # At least one layer
        self.drop = nn.Dropout(p=0.5)
        self.linear_lvl1 = nn.Linear(
            self.out_channels * self.expansion, self.out_channels
        )
        self.relu_lv1 = nn.ReLU(inplace=False)
        self.softmax_reg1 = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        """Forward propagation of pretrained ResNet-50."""
        x = self.model_ft(x)

        x = self.drop(x)  # Dropout to add regularization

        level_1 = self.softmax_reg1(self.relu_lv1(self.linear_lvl1(x)))
        # level_1 = nn.Softmax(level_1)

        return level_1


def load_models(device, localisation_model_path, binary_model_path, order_model_path, order_threshold_path, regional_model_path, regional_map_path):

    # Load the localisation model
    weights_path = localisation_model_path

    model_loc = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # 1 class (object) + background
    in_features = model_loc.roi_heads.box_predictor.cls_score.in_features
    model_loc.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    model_loc.load_state_dict(state_dict)
    model_loc = model_loc.to(device)
    model_loc.eval()

    # Load the binary model
    weights_path = binary_model_path
    classification_model = timm.create_model(
        "tf_efficientnetv2_b3", num_classes=num_classes, weights=None
    )
    classification_model = classification_model.to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    classification_model.load_state_dict(state_dict)
    classification_model.eval()

    # Load the order model
    savedWeights = order_model_path
    thresholdFile = order_threshold_path
    order_data_thresholds = pd.read_csv(thresholdFile)
    order_labels = order_data_thresholds["ClassName"].to_list()
    num_classes = len(order_labels)
    model_order = ResNet50_order(num_classes=num_classes)
    model_order.load_state_dict(torch.load(savedWeights, map_location=device, weights_only=True))
    model_order = model_order.to(device)
    model_order.eval()

    # Load the species classifier model
    weights = regional_model_path
    species_category_map = json.load(
        open(
            regional_map_path
        )
    )
    num_classes = len(species_category_map)
    species_model = Resnet50_species(num_classes=num_classes)
    species_model = species_model.to(device)
    checkpoint = torch.load(weights, map_location=device, weights_only=True)
    # The model state dict is nested in some checkpoints, and not in others
    state_dict = checkpoint.get("model_state_dict") or checkpoint
    species_model.load_state_dict(state_dict)
    species_model.eval()

    return ({
        'localisation_model': model_loc,
        'classification_model': classification_model,
        'species_model': species_model,
        'species_model_labels': species_category_map,
        'order_model': model_order,
        'order_model_thresholds': order_data_thresholds,
        'order_model_labels': order_labels
    })
