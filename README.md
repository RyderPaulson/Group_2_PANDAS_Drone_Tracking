# PANDAS: Point-based Autonomous Networked Drone Aiming System
Siman Zhang | James Lewis | Shree Singhal | Ryder Paulson | Patrick Donnelly | Burke Havranek

A laser-based tracking and charging system designed to enable continuous in-flight drone operation.

## Installation Guide

1. Clone the repository.

```
git clone https://github.com/RyderPaulson/Group_2_PANDAS_Drone_Tracking
cd Group_2_PANDAS_Drone_Tracking
```

2. Create a v3.12.11 Python virtual environment. Conda is recommended but not required.
3. Install CUDA.

```
# check if cuda is installed
nvcc --version
```

If it is not installed, install the appropriate CUDA 12.6 version for your system from [here](https://developer.nvidia.com/cuda-12-6-0-download-archive). Set the CUDA_HOME environmental variable to your CUDA instillation.

4. Clone and install CoTracker3.

```
git clone https://github.com/facebookresearch/co-tracker
cd co-tracker
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard
mkdir weights
cd weights
# download the online (multi window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
# download the online (sliding window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/baseline_online.pth
cd ..
cd ..
```

5. Clone and install GroundingDINO.

```
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
```

6. Implement [this change](https://github.com/IDEA-Research/GroundingDINO/pull/415) from GroundingDINO
7. Finish the install of GroundingDINO

```
pip install -e .
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```



6. Download the tiny weights from our Google Drive [groundingdino_swint_ogc](https://drive.google.com/file/d/1JuvtEhrE4oPX_5h5dwdAwMy35T0FAJLR/view?usp=drive_link) into the weights folder.
7. Uninstall PyTorch and bundled packages.

```
pip uninstall torch torchvision
```

8. Navigate to [PyTorch's website](https://pytorch.org/) and install get the correct command for installing PyTorch with CUDA 12.6 on your system.
9. Install remaining required Python packages. 

TODO



## Requirements
### Core
1. PyTorch > 2.0.0
2. CV2 Latest
3. CUDA 12.6
4. [CoTracker3](https://github.com/facebookresearch/co-tracker)
5. [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

### Jetson Specific
TODO Check on Jetson board for specific requirements. 