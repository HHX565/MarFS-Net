# <div align="center">

# 

# \# 🌊 MarFS-Net

# 

# \### Maritime Obstacle Segmentation Framework based on Frequency-Domain Purification and Spatial Calibration

# 

# \[!\[Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)

# \[!\[PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C.svg)](https://pytorch.org/)

# \[!\[CUDA](https://img.shields.io/badge/CUDA-12.8-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

# \[!\[Task](https://img.shields.io/badge/Task-Maritime%20Segmentation-brightgreen.svg)](https://github.com/HHX565/MarFS-Net)

# 

# Official PyTorch implementation of \*\*MarFS-Net\*\*.

# 

# </div>

# 

# \---

# 

# \## 📖 Abstract

# 

# Maritime obstacle segmentation is an essential component of autonomous navigation for unmanned surface vehicles. However, complex maritime environments contain strong reflections, wave glints, shadows, motion-related structural degradation, blurred boundaries, and distant small obstacles, which can considerably reduce segmentation accuracy.

# 

# To address these challenges, we propose \*\*MarFS-Net\*\*, a single-frame RGB maritime obstacle segmentation framework based on frequency-domain purification and spatial calibration.

# 

# MarFS-Net contains three key components:

# 

# \- \*\*Wavelet-Guided Polarized Spatial Aggregation (W-PSA)\*\* separates low-frequency structural information from directional high-frequency details and suppresses unreliable responses caused by reflections, glare, and shadows.

# \- \*\*Hybrid Feature Modulation for Degradation (HFMD)\*\* combines local differential compensation, KAN-based nonlinear modulation, global structural calibration, and cross-branch gated fusion.

# \- \*\*Stage-wise Multi-scale Deep Supervision (MDS)\*\* strengthens weak and small-target responses during early training and focuses optimization on the final fused prediction during late training.

# 

# On the complex-sea-state evaluation subset constructed from the public LaRS dataset, MarFS-Net achieves an \*\*mIoU of 0.7085\*\*, an \*\*mDice of 0.7604\*\*, an \*\*mRecall of 0.8069\*\*, an \*\*mBIoU of 0.5976\*\*, and an \*\*mHD95 of 22.4301\*\*.

# 

# <p align="center">

# &#x20; <img src="./MarFS-Net.png" width="95%" alt="MarFS-Net Architecture">

# </p>

# 

# <p align="center">

# &#x20; <em>The overall architecture of MarFS-Net.</em>

# </p>

# 

# \---

# 

# \## ⚙️ Implementation

# 

# \### 🖥️ Experimental Environment

# 

# The experiments reported in the paper were conducted using the following environment:

# 

# | Item | Configuration |

# |---|---|

# | Operating system | Windows 11 |

# | CPU | AMD Ryzen 9 8945HX with Radeon Graphics |

# | GPU | NVIDIA GeForce RTX 5060 Laptop GPU |

# | GPU memory | 8 GB |

# | Python | 3.10 |

# | PyTorch | 2.9.0 |

# | CUDA | 12.8 |

# 

# \---

# \### 🛠️ Installation

# 

# Clone the repository:

# 

# ```bash

# git clone https://github.com/HHX565/MarFS-Net.git

# cd MarFS-Net

# ```

# 

# Create and activate the environment:

# 

# ```bash

# conda create -n marfsnet python=3.10 -y

# conda activate marfsnet

# ```

# 

# Install PyTorch with CUDA 12.8:

# 

# ```bash

# pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \\

# &#x20; --index-url https://download.pytorch.org/whl/cu128

# ```

# 

# Install the remaining dependencies:

# 

# ```bash

# pip install numpy pandas scipy scikit-learn opencv-python \\

# &#x20; pillow albumentations tqdm pyyaml yacs matplotlib sympy

# ```

# 

# Verify the environment:

# 

# ```bash

# python -c "import torch; print('PyTorch:', torch.\_\_version\_\_); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is\_available())"

# ```

# 

# \### 📋 Training Configuration

# 

# | Parameter | Setting |

# |---|---|

# | Input size | 256 × 256 |

# | Input channels | 3 |

# | Number of classes | 8 |

# | Training/validation split | 7:3 |

# | Random seed | 3407 |

# | Epochs | 300 |

# | Batch size | 4 |

# | Gradient accumulation | 2 |

# | Effective batch size | 8 |

# | Optimizer | AdamW |

# | Maximum learning rate | 1 × 10⁻³ |

# | Weight decay | 1 × 10⁻² |

# | Scheduler | OneCycleLR |

# | Loss function | DiceCE |

# | Cross-entropy/Dice weights | 0.5 / 0.5 |

# | Label smoothing | 0.1 |

# | Dice smoothing constant | 1.0 |

# | Ignore label | 255 |

# | Mixed precision | AMP |

# | Model averaging | EMA |

# | Input resizing | 256 × 256 |

# | Training augmentation | Horizontal flip, vertical flip, normalization |

# | Validation preprocessing | Resize and normalization |

# | Evaluation metrics | mIoU, mDice, mRecall, mBIoU, mHD95 |

# 

# \---



# \## 📂 Data Format

# 

# Organize the dataset using the following directory structure:

# 

# ```text

# inputs/

# └── <dataset\_name>/

# &#x20;   ├── images/

# &#x20;   │   ├── 001.png

# &#x20;   │   ├── 002.png

# &#x20;   │   ├── 003.png

# &#x20;   │   └── ...

# &#x20;   │

# &#x20;   └── masks/

# &#x20;       └── 0/

# &#x20;           ├── 001.png

# &#x20;           ├── 002.png

# &#x20;           ├── 003.png

# &#x20;           └── ...

# ```

# 

# Example:

# 

# ```text

# inputs/

# └── custom/

# &#x20;   ├── images/

# &#x20;   │   ├── 001.png

# &#x20;   │   ├── 002.png

# &#x20;   │   └── 003.png

# &#x20;   │

# &#x20;   └── masks/

# &#x20;       └── 0/

# &#x20;           ├── 001.png

# &#x20;           ├── 002.png

# &#x20;           └── 003.png

# ```

# 

# 

# The masks should be stored as single-channel class-index images:

# 

# | Pixel value | Class |

# |---:|---|

# | 0 | Background |

# | 1 | Water surface |

# | 2 | Sky |

# | 3 | Navigating vessel |

# | 4 | Moored or stationary vessel |

# | 5 | Floating debris |

# | 6 | Animal |

# | 7 | Human |

# | 255 | Ignored label |

# 

# \---

# 

# \## 🚀 Training and Validation

# 

# \- \*\*Train the model\*\*

# 

# ```bash

# python train.py

# 

# \- \*\*Evaluate the model\*\*

# 

# Before evaluation, set the checkpoint and dataset paths in `test.py`:

# 

# ```python

# 'model\_path': './outputs/MarFS-Net/model\_best.pth',

# 'dataset\_dir': './inputs/custom'

# ```

# 

# Then run:

# 

# ```bash

# python test.py

# ```

# 

# The evaluation script reports:

# 

# ```text

# mIoU

# mDice

# mRecall

# mPrecision

# mBIoU

# mHD95

# ```

# 

# The per-image evaluation results are saved as:

# 

# ```text

# outputs/MarFS-Net/final\_test\_results\_with\_biou.csv

# ```

# 

# \---

# 

# 

# <div align="center">

# 

# \### ⭐ If this repository is helpful to your research, please consider giving it a star!

# 

# </div>

