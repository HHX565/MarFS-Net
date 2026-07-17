<div align="center">
  <h1>🌊 MarFS-Net</h1>
  <p><strong>Maritime Obstacle Segmentation Framework based on Frequency-Domain Purification and Spatial Calibration</strong></p>
  <p>Official PyTorch implementation of MarFS-Net</strong></p>
</div>

## 📖 Abstract

Maritime obstacle segmentation is essential for the autonomous navigation of unmanned surface vehicles (USVs). However, complex maritime environments contain strong reflections, wave glints, shadows, motion-related structural degradation, blurred boundaries, and distant small obstacles, which can substantially reduce segmentation accuracy.

To address these challenges, we propose **MarFS-Net**, a single-frame RGB maritime obstacle segmentation framework based on frequency-domain purification and spatial calibration.

MarFS-Net contains three principal components:

- **Wavelet-Guided Polarized Spatial Aggregation (W-PSA)** separates stable low-frequency structures from directional high-frequency details and suppresses unreliable responses caused by reflections, glare, wave glints, and shadows.
- **Hybrid Feature Modulation for Degradation (HFMD)** combines local differential compensation, KAN-based nonlinear mapping, global structural calibration, and cross-branch gated fusion.
- **Stage-wise Multi-scale Deep Supervision (MDS)** strengthens weak and small-target responses during early training and focuses late-stage optimization on the final fused prediction.

On the complex-sea-state evaluation subset constructed from the public LaRS dataset, MarFS-Net achieves an **mIoU of 0.7085**, an **mDice of 0.7604**, an **mRecall of 0.8069**, an **mBIoU of 0.5976**, and an **mHD95 of 22.4301**.

<p align="center">
  <img width="1902" height="991" alt="MarFS-Net" src="https://github.com/user-attachments/assets/4e32514e-9097-452a-9ad1-a37a719c5e0f" />
</p>
<p align="center">
  <em>Overall architecture of the proposed MarFS-Net.</em>
</p>

## ⚙️ Implementation

### Experimental Environment

The experiments reported in the paper were conducted using the following environment:

```text
Operating system : Windows 11
CPU              : AMD Ryzen 9 8945HX with Radeon Graphics
GPU              : NVIDIA GeForce RTX 5060 Laptop GPU
GPU memory       : 8 GB
Python           : 3.10
PyTorch          : 2.9.0
CUDA             : 12.8
```

### Installation

- Clone this repository:

```bash
git clone https://github.com/SilasHan/MarFS-Net.git
cd MarFS-Net
```

- Create and activate the Conda environment:

```bash
conda create -n marfsnet python=3.10 -y
conda activate marfsnet
```

- Upgrade pip:

```bash
python -m pip install --upgrade pip
```

- Install PyTorch with CUDA 12.8:

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

- Install the remaining dependencies:

```bash
pip install numpy pandas scipy scikit-learn opencv-python pillow albumentations tqdm pyyaml yacs matplotlib sympy
```

- Verify the environment:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

### Training Configuration

The main experimental configuration is summarized below:

```text
Model class                : MSHNet_Official
Input size                 : 256 × 256
Input channels             : 3
Number of classes          : 8
Training/validation split  : 7:3
Random seed                : 3407
Epochs                     : 300
Batch size                 : 4
Gradient accumulation      : 2
Effective batch size       : 8
Optimizer                  : AdamW
Maximum learning rate      : 1 × 10⁻³
Weight decay               : 1 × 10⁻²
Learning-rate scheduler    : OneCycleLR
Loss function              : DiceCE
Cross-entropy/Dice weights : 0.5 / 0.5
Label smoothing            : 0.1
Dice smoothing constant    : 1.0
Ignore label               : 255
Training strategy          : AMP and EMA
```

The training augmentation and validation preprocessing are:

```text
Training   : Resize, random horizontal flip, random vertical flip, normalization
Validation : Resize and normalization
Metrics    : mIoU, mDice, mRecall, mBIoU and mHD95
```
## 📂 Data Format

- Arrange the dataset using the following directory structure:

```text
inputs/
└── <dataset_name>/
    ├── images/
    │   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   └── ...
    └── masks/
        └── 0/
            ├── 001.png
            ├── 002.png
            ├── 003.png
            └── ...
```

- Masks should be stored as single-channel class-index images:

```text
0   : Background
1   : Water surface
2   : Sky
3   : Navigating vessel
4   : Moored or stationary vessel
5   : Floating debris
6   : Animal
7   : Human
255 : Ignored label
```
## 🚀 Training and Validation

### Train the Model

- Start training:

```bash
python train.py
```

- The dataset is automatically divided into training and validation sets using:

```text
Training/validation ratio : 7:3
Random seed               : 3407
```

- The best checkpoint and training records are saved under:

```text
outputs/
└── MarFS-Net/
    ├── model_best.pth
    ├── config.yml
    └── training_log.csv
```

The best checkpoint is selected according to validation IoU:

```text
outputs/MarFS-Net/model_best.pth
```

The training log contains:

```text
Epoch
Train_Loss
Train_IoU
Val_Loss
Val_IoU
Learning_Rate
```

### Evaluate the Model

- Set the model and dataset paths in `test.py`:

```python
'model_path': './outputs/MarFS-Net/model_best.pth',
'dataset_dir': './inputs/custom'
```

- Run evaluation:

```bash
python test.py
```

- The evaluation script reports:

```text
mIoU
mDice
mRecall
mPrecision
mBIoU
mHD95
```

- Per-image evaluation results are saved to:

```text
outputs/MarFS-Net/final_test_results_with_biou.csv
```

<p align="center">
  <strong>⭐ If this repository is helpful to your research, please consider giving it a star!</strong>
</p>