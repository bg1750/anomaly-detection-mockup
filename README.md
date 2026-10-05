# Anomaly Detection

Reconstruction-based visual anomaly detection with a convolutional autoencoder,
built in PyTorch. The model is trained only on "normal" frames, so it learns to
rebuild them with low error. When a frame contains something it never saw in
training, the reconstruction error rises; if it stays high for several frames in
a row, the frame is flagged as an anomaly.

This is the generic version of instrument-inspection work I can't share publicly.
The code and the trained weights are here; no source imagery or data is included.
Point it at your own webcam or a folder of your own frames to run it.

## How it works

1. Each frame is resized to 512x512 and passed through the autoencoder
   (`model.py`): two convolutional down-sampling stages, then two transposed-conv
   stages back up to the original size.
2. The mean-squared error between the input frame and its reconstruction is the
   anomaly score.
3. A frame is counted as anomalous when that score is at or above a threshold.
   When the anomalous count reaches a set number of consecutive frames, an
   anomaly is reported. This consecutive-frame rule suppresses one-off spikes.

## Run

```
pip install -r requirements.txt

# live webcam
python detect.py --camera 0

# or over a local folder of frames (nothing from it is committed)
python detect.py --images ./frames --threshold 0.002 --consecutive 5
```

Press `q` to quit. Three windows show the input, the reconstruction, and the
difference between them.

## Train your own

```
python train.py --data ./frames --epochs 30
```

`--data` is a folder of normal frames. Training frames stay local.

## Files

- `model.py` — the autoencoder architecture
- `detect.py` — live reconstruction + MSE thresholding + consecutive-frame rule
- `train.py` — trains the autoencoder on a folder of normal frames
- `conv_autoencoder.pth` — trained weights (3-channel, 512x512)
