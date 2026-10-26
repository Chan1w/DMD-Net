"""Image datasets for the DJSCC + DMD-Net pipeline.

Splits are done at the *original-image* level: for VOC the official
``ImageSets/Main/{train,val}.txt`` are used; a loose image folder (e.g. Kodak24)
is treated as an external test set.  The same image never appears in two splits
under different channel conditions (spec 9 / stage 1).
"""

from __future__ import annotations

import os
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VOCImageDataset(Dataset):
    """PASCAL VOC 2012 images keyed by an official ImageSets split file."""

    def __init__(self, data_root: str, split: str = "train", image_size: int = 384):
        self.data_root = data_root
        self.image_size = image_size
        self.img_dir = os.path.join(data_root, "JPEGImages")
        split_file = os.path.join(data_root, "ImageSets", "Main", f"{split}.txt")
        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # -> [0, 1]
        ])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        path = os.path.join(self.img_dir, img_id + ".jpg")
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return {"image": x, "id": img_id, "path": path}


class ImageFolderDataset(Dataset):
    """Generic loose-image dataset (Kodak24, CLIC, DIV2K, ...), sorted by name."""

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    def __init__(self, data_root: str, image_size: int = 384):
        self.data_root = data_root
        self.image_size = image_size
        self.files = sorted(
            name for name in os.listdir(data_root)
            if name.lower().endswith(self.IMG_EXTS)
        )
        self.ids = [os.path.splitext(name)[0] for name in self.files]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # -> [0, 1]
        ])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        path = os.path.join(self.data_root, self.files[idx])
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return {"image": x, "id": img_id, "path": path}


def to_vae_input(image: torch.Tensor) -> torch.Tensor:
    """Map [0,1] image tensor to the SD-VAE input range [-1, 1]."""
    return image * 2.0 - 1.0


def from_vae_output(image: torch.Tensor) -> torch.Tensor:
    """Map SD-VAE output [-1,1] back to [0,1] and clamp."""
    return ((image + 1.0) / 2.0).clamp(0.0, 1.0)
