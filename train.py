"""Train the convolutional autoencoder on a folder of "normal" frames.

Point --data at a directory of images that represent the normal scene. The
model learns to reconstruct them; anything it has not seen then reconstructs
poorly and shows up as a high error at detection time. Training frames stay
local and are never committed.

Usage:
    python train.py --data ./frames --epochs 30 --batch-size 16
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import Autoencoder

IMG_RES = (512, 512)


class FrameDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.images[idx])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image  # input and reconstruction target are the same


def train(model, loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="folder of normal frames")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--out", default="conv_autoencoder.pth")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize(IMG_RES),
        transforms.ToTensor(),
    ])
    dataset = FrameDataset(args.data, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    model.to(device)

    train(model, loader, criterion, optimizer, args.epochs, device)
    torch.save(model.state_dict(), args.out)
    print("Saved model to", args.out)


if __name__ == "__main__":
    main()
