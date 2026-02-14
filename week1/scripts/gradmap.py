import os

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models


# global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.getcwd()
SHAPE = (224, 224)

sample = "sea_lion.png"
SAMPLE_PATH = os.path.join(BASE_DIR, "data", sample)


def get_model(device: torch.device = "cpu") -> nn.Module:
    model = models.resnet18(pretrained=True).to(device).eval()
    return model


def read_sample(path: str) -> np.ndarray:
    sample = cv2.imread(path)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    return sample


def preprocess_image(img: np.ndarray, shape: tuple = (224, 224)) -> np.ndarray:
    """preprocess image to input shape and normalize to norm distribution"""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img_resized = cv2.resize(img, shape)
    img_normalized = img_resized / 255.0
    return (img_normalized - mean) / std


def to_tensor(img: np.ndarray, device: torch.device = "cpu") -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)


def get_activation_map(feat_tensor: torch.Tensor):
    # calc mean across channels, ~get single 2D activation map
    act_map = torch.mean(feat_tensor[0], dim=0).cpu().numpy()

    # normalize to [0, 255]
    act_map = np.maximum(act_map, 0)
    act_map /= np.max(act_map) + 1e-5
    return (act_map * 255.).astype(np.uint8)


def plot_activation_maps(maps: list[np.ndarray], savefig: bool = False):
    _, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, ax in enumerate(axes):
        ax.imshow(maps[i], cmap="magma")
        ax.axis("off")

    plt.tight_layout()
    if savefig:
        plt.savefig("assets/feature_maps.png")
    plt.show()


@torch.no_grad()
def extract_feature_maps(model: nn.Module, input_tensor: torch.Tensor) -> list[torch.Tensor]:
    x = model.maxpool(model.relu(model.bn1(model.conv1(input_tensor))))
    f1 = model.layer1(x)
    f2 = model.layer2(f1)
    f3 = model.layer3(f2)
    f4 = model.layer4(f3)
    return [f1, f2, f3, f4]


def pipeline(sample_path: str, device: torch.device = "cpu"):
    image = read_sample(sample_path)
    preprocessed = preprocess_image(image, SHAPE)
    input_tensor = to_tensor(preprocessed, device)

    model = get_model(device)
    maps = [get_activation_map(t) for t in extract_feature_maps(model, input_tensor)]
    plot_activation_maps(maps[:3])


if __name__ == "__main__":
    pipeline(SAMPLE_PATH, DEVICE)