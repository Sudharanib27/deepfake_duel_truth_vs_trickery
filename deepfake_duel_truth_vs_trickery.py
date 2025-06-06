# -*- coding: utf-8 -*-
"""Deepfake_Duel_Truth_vs_Trickery.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uwaiD2sRLWRj7aUdVPE3-CpJN-vjAHtl
"""

# !pip install torchcam
!pip install numpy==1.24.4

import zipfile
import os
import random
import numpy as np
import torch
from tqdm import tqdm

# Function to set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

print("Extracting dataset from Google Drive...")
archive_path = '/content/drive/MyDrive/ArtiFact_240K.zip'

# Open the zip file and get the list of files
with zipfile.ZipFile(archive_path, 'r') as zip_ref:
    file_list = zip_ref.namelist()

    # Create a tqdm progress bar for the file extraction process
    with tqdm(total=len(file_list), desc="Extracting files") as pbar:
        for file in file_list:
            zip_ref.extract(file, '/content')
            pbar.update(1)

print("Dataset extracted successfully!")

base_dir = '/content/ArtiFact_240K/'
print(f"Exploring dataset structure at {base_dir}")

import os
from torch.utils.data import Dataset
from PIL import Image

class ArtiFactDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.class_map = {'human_faces': 0, 'animals': 1, 'vehicles': 2}

        for label_name in ['real', 'fake']:
            label = 1 if label_name == 'real' else 0
            for cls in os.listdir(os.path.join(root_dir, label_name)):
                cls_dir = os.path.join(root_dir, label_name, cls)
                for img_name in os.listdir(cls_dir):
                    self.samples.append({
                        'img_path': os.path.join(cls_dir, img_name),
                        'label': label,
                        'class': self.class_map[cls]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['img_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, sample['label'], sample['class']

import os
from collections import defaultdict

base_dir = "/content/ArtiFact_240K/train"
label_map = {"real": 1, "fake": 0}
class_map = {'human_faces': 0, 'animals': 1, 'vehicles': 2}

# Count storage
counts = defaultdict(int)

# Count files in each class per label
for label_name in ['real', 'fake']:
    for cls in os.listdir(os.path.join(base_dir, label_name)):
        cls_dir = os.path.join(base_dir, label_name, cls)
        n = len(os.listdir(cls_dir))
        counts[(label_name, cls)] = n

# Print result
print(f"{'Label':<10} {'Class':<15} {'Count'}")
print("-" * 40)
for (label, cls), count in sorted(counts.items()):
    print(f"{label:<10} {cls:<15} {count}")

"""Although the dataset is balanced in terms of real vs. fake within each class, the total number of samples per class is imbalanced:

* human_faces: 84,000 images
* vehicles: 58,798 images
* animals: 25,200 images

This imbalance can cause the model to bias toward predicting human_faces, as it has more representation during training.

To handle this, we apply class-weighted CrossEntropyLoss, where classes with fewer samples are given higher importance. This helps the model treat all classes fairly during learning.
"""

from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# resize + normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Training dataset
train_dataset = ArtiFactDataset("/content/ArtiFact_240K/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# Validation dataset
val_dataset = ArtiFactDataset("/content/ArtiFact_240K/validation", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class MultiTaskEffNet(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiTaskEffNet, self).__init__()

        # Load pretrained EfficientNetV2 backbone
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()

        # Feature output size from backbone
        feature_dim = self.backbone.classifier.in_features if hasattr(self.backbone.classifier, 'in_features') else 1280

        # Real/Fake head (binary)
        self.real_fake_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # output logit for BCEWithLogitsLoss
        )

        # Class head (multiclass)
        self.class_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)  # output logits for CrossEntropyLoss
        )

    def forward(self, x):
        features = self.backbone(x)
        real_fake_out = self.real_fake_head(features).squeeze(1)
        class_out = self.class_head(features)
        return real_fake_out, class_out

model = MultiTaskEffNet().to(device)

# Loss functions
binary_loss_fn = nn.BCEWithLogitsLoss()
class_weights = torch.tensor([1/84000, 1/25200, 1/58798])
class_weights = class_weights / class_weights.sum()
class_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, binary_loss_fn, class_loss_fn, device):
    model.train()
    total_loss, total_correct_bin, total_correct_cls = 0, 0, 0
    total_samples = 0

    for images, labels_bin, labels_cls in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels_bin = labels_bin.to(device).float()
        labels_cls = labels_cls.to(device)

        optimizer.zero_grad()
        out_bin, out_cls = model(images)

        loss_bin = binary_loss_fn(out_bin, labels_bin)
        loss_cls = class_loss_fn(out_cls, labels_cls)
        loss = loss_bin + loss_cls

        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item() * images.size(0)
        total_correct_bin += ((torch.sigmoid(out_bin) > 0.5).int() == labels_bin.int()).sum().item()
        total_correct_cls += (out_cls.argmax(1) == labels_cls).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct_bin / total_samples, total_correct_cls / total_samples

def evaluate(model, loader, binary_loss_fn, class_loss_fn, device):
    model.eval()
    total_loss, total_correct_bin, total_correct_cls = 0, 0, 0
    total_samples = 0

    with torch.no_grad():
        for images, labels_bin, labels_cls in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            labels_bin = labels_bin.to(device).float()
            labels_cls = labels_cls.to(device)

            out_bin, out_cls = model(images)

            loss_bin = binary_loss_fn(out_bin, labels_bin)
            loss_cls = class_loss_fn(out_cls, labels_cls)
            loss = loss_bin + loss_cls

            total_loss += loss.item() * images.size(0)
            total_correct_bin += ((torch.sigmoid(out_bin) > 0.5).int() == labels_bin.int()).sum().item()
            total_correct_cls += (out_cls.argmax(1) == labels_cls).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct_bin / total_samples, total_correct_cls / total_samples

from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import os

def visualize_gradcam(model, dataset, index=0, save_dir="gradcam_outputs", device='cuda'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    image, label_bin, label_cls = dataset[index]
    input_tensor = image.unsqueeze(0).to(device)

    # Attach GradCAM to a suitable conv layer
    target_layer = model.backbone.features[-1]
    cam_extractor = GradCAM(model, target_layer=target_layer)

    # Forward pass
    output_bin, _ = model(input_tensor)
    if output_bin.ndim == 1:
      output_bin = output_bin.unsqueeze(0)

    # Use the predicted class or a known index
    class_idx = output_bin.argmax().item()
    activation_map = cam_extractor(class_idx, output_bin)[0].cpu()

    # Visualize
    overlay = overlay_mask(to_pil_image(image), to_pil_image(activation_map, mode='F'), alpha=0.5)
    overlay.save(f"{save_dir}/sample_{index}_real_fake.png")

import csv

log_file = "training_log.csv"
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc_bin', 'val_acc_bin', 'train_acc_cls', 'val_acc_cls'])

    best_val_loss = float('inf')

    for epoch in range(1, 11):
        train_loss, train_acc_bin, train_acc_cls = train_one_epoch(model, train_loader, optimizer, binary_loss_fn, class_loss_fn, device)
        val_loss, val_acc_bin, val_acc_cls = evaluate(model, val_loader, binary_loss_fn, class_loss_fn, device)

        # Log to CSV
        writer.writerow([epoch, train_loss, val_loss, train_acc_bin, val_acc_bin, train_acc_cls, val_acc_cls])

        print(f"Epoch {epoch}")
        print(f"  Train Loss: {train_loss:.4f}, Real/Fake Acc: {train_acc_bin:.4f}, Class Acc: {train_acc_cls:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Real/Fake Acc: {val_acc_bin:.4f}, Class Acc: {val_acc_cls:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

torch.save(model.state_dict(), "best_model.pt")

visualize_gradcam(model, val_dataset, index=10, device=device)

from torch.utils.data import Dataset
from PIL import Image
import os

class ArtiFactTestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.image_names = sorted(os.listdir(test_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.test_dir, self.image_names[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_names[idx]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_dataset = ArtiFactTestDataset("/content/ArtiFact_240K/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = MultiTaskEffNet().to(device)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

from torchvision import transforms

tta_transforms = [
    transforms.Compose([]),  # identity (no transform)
    transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
    transforms.Compose([transforms.RandomRotation(degrees=15)]),
    transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]),
]

predictions = []
class_map_rev = {0: 'human_faces', 1: 'animals', 2: 'vehicles'}

with torch.no_grad():
    for images, image_names in tqdm(test_loader, desc="Predicting with TTA"):
        images = images.to(device)
        batch_size = images.size(0)

        all_bin_preds = []
        all_cls_logits = []

        # Run each TTA transform
        for tta in tta_transforms:
            tta_images = torch.stack([tta(img.cpu()) for img in images])  # apply TTA
            tta_images = tta_images.to(device)

            out_bin, out_cls = model(tta_images)

            all_bin_preds.append(torch.sigmoid(out_bin))      # binary probs
            all_cls_logits.append(out_cls)                    # raw logits for softmax

        # Average predictions
        avg_bin = torch.mean(torch.stack(all_bin_preds), dim=0)
        avg_cls_logits = torch.mean(torch.stack(all_cls_logits), dim=0)

        # Final predictions
        final_bin = (avg_bin > 0.5).int().cpu().numpy()
        final_cls = avg_cls_logits.argmax(1).cpu().numpy()

        for name, label, cls in zip(image_names, final_bin, final_cls):
            predictions.append([name, int(label), class_map_rev[int(cls)]])

import pandas as pd

# Convert predictions to DataFrame
df = pd.DataFrame(predictions, columns=["image", "label", "class"])

# Save to test.csv
df.to_csv("test.csv", index=False)
print("Saved predictions to test.csv")