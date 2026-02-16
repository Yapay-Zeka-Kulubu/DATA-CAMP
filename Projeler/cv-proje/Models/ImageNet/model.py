import torch.nn as nn
from torchvision import models
import config

def build_model():
    if config.MODEL_NAME == "resnet18":
        model = models.resnet18(pretrained=config.PRETRAINED)
    elif config.MODEL_NAME == "resnet50":
        model = models.resnet50(pretrained=config.PRETRAINED)
    else:
        raise ValueError("Desteklenmeyen model")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, config.NUM_CLASSES)

    return model
