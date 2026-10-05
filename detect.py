"""Live reconstruction-based anomaly detection.

Runs the trained convolutional autoencoder on a video source frame by frame.
Each frame is reconstructed, the mean-squared reconstruction error is measured,
and a frame is counted as anomalous when that error stays above a threshold for
several frames in a row.

Sources:
    --camera N        use webcam / capture device index N (default: 0)
    --images DIR      run over a folder of image frames instead of a camera
                      (kept local; nothing from such a folder is committed)

Usage:
    python detect.py --camera 0
    python detect.py --images ./frames --threshold 0.002 --consecutive 5
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from model import Autoencoder

DIM = (512, 512)


def to_tensor(frame_bgr):
    """Match the preprocessing the bundled weights were trained/run with."""
    frame = cv2.resize(frame_bgr, DIM)
    frame = frame / 255.0
    return torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()


def frames_from_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open capture device {index}.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def frames_from_dir(path):
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(glob.glob(os.path.join(path, ext)))
    files.sort()
    if not files:
        raise SystemExit(f"No image frames found in {path}.")
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            yield img


def detect(frames, model, criterion, threshold, consecutive):
    anomaly_counter = 0
    for frame in frames:
        tensor_frame = to_tensor(frame)

        with torch.no_grad():
            reconstructed = model(tensor_frame)

        loss = criterion(reconstructed, tensor_frame)

        recon = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
        original = tensor_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        diff = np.abs(original - recon)

        cv2.imshow("Original", original)
        cv2.imshow("Reconstructed", recon)
        cv2.imshow("Difference", diff / (diff.max() + 1e-8))

        if loss.item() >= threshold:
            anomaly_counter += 1
        else:
            anomaly_counter = 0

        status = "ANOMALY" if anomaly_counter >= consecutive else "normal"
        print(f"loss={loss.item():.6f}  streak={anomaly_counter}  {status}")

        if anomaly_counter >= consecutive:
            print("Anomaly detected!")
            anomaly_counter = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="capture device index")
    parser.add_argument("--images", help="folder of frames to run over instead of a camera")
    parser.add_argument("--weights", default="conv_autoencoder.pth", help="model weights")
    parser.add_argument("--threshold", type=float, default=0.002, help="reconstruction-error threshold")
    parser.add_argument("--consecutive", type=int, default=5, help="anomalous frames in a row to flag")
    args = parser.parse_args()

    model = Autoencoder()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    criterion = nn.MSELoss()

    frames = frames_from_dir(args.images) if args.images else frames_from_camera(args.camera)
    detect(frames, model, criterion, args.threshold, args.consecutive)


if __name__ == "__main__":
    main()
