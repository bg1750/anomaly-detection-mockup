"""Convolutional autoencoder used for reconstruction-based anomaly detection.

The network compresses each frame through two convolutional stages and then
reconstructs it. It is trained only on "normal" frames, so it learns to rebuild
them with low error. When a frame contains something the model never saw during
training, the reconstruction error rises and the frame is flagged.

This is the same architecture the bundled weights (conv_autoencoder.pth) were
trained with: 3-channel input, 512x512 frames.
"""

import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
