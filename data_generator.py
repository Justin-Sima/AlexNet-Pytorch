import os
import pandas as pd
import torch
import torchvision

# Resizing, random crop, and horizontal flip with probability 0.5.
alexnet_preprocessing = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.RandomCrop(224),
    torchvision.transforms.RandomHorizontalFlip(0.5)
])

class AlexNetImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_directory, label_file):
        self.image_labels = pd.read_csv(label_file)
        self.image_directory = image_directory
        self.transform = alexnet_preprocessing

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_directory, self.image_labels.iloc[idx, 0])
        image = torchvision.io.read_image(img_path)
        image_transformed = image = self.transform(image)
        label = self.image_labels.iloc[idx, 1]

        return image_transformed, label

def create_alexnet_dataloader(image_directory, label_file, batch_size=256):
    data_loader = torch.utils.data.DataLoader(
        dataset=AlexNetImageDataset(image_directory=image_directory, label_file=label_file),
        batch_size=batch_size,
        shuffle=True
    )

    return data_loader
