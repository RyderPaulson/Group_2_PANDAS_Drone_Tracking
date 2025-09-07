# Simulation Suite for PANDAS

## Getting Started

To get started using this repository, we will need to first do the following. It is assumed that MATLAB R2025a is already installed on your system.

1. Install Unreal Engine
2. If using a CUDA enabled device, install the CUDA toolkit. Running co-tracker on the CPU is not recommended
3. Cloning the CoTracker 3 Github repository.
4. Create a new Python environment
   - Install the necessary packages
5. Downloading supplemental assets



### Installing Unreal Engine

Unreal Engine is developed by Epic Games, so we will first have install the Epic Games launcher linked [here](https://store.epicgames.com/en-US). Once the launcher is installed and you have logged in, navigate to the Unreal Engine pane on the left side of the application. Go to the Library section on the top of the application. Click on the + icon next to "Engine Versions" and download **Unreal Engine Version 5.3.2**. Once Unreal Engine is installed, it should appear under the Engine Versions (see image below).

![Unreal Engine once installed using Epic Games Launcher](media/ue_in_launcher.png)

You have now successfully installed Unreal Engine.



### Installing CUDA Toolkit

We will now be installing version 12.9 of the CUDA Toolkit. Navigate to [this](https://developer.nvidia.com/cuda-12-9-0-download-archive) link and download the appropriate version for your system. Run the downloaded installer  and follow the prompts (I already have it installed so I cannot easily run through the process to get images). To ensure that it is properly installed on your system run the following in your terminal.

```
nvcc --version
```

It should return the following.

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:24:01_Pacific_Daylight_Time_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0
```



### Setting Up CoTracker

In the root directory of your project, run the following command.

```
git clone https://github.com/facebookresearch/co-tracker
```

Please note that while this procedure is sourced from the [CoTracker3 Github installation instructions](https://github.com/facebookresearch/co-tracker), it is not required to following the preceding steps installing specific Python packages. That will be accomplished in the next section of this guide. 

Next we will download the necessary checkpoints for running CoTracker3 in online mode. From the root directory of your project (not the root directory of co-tracker), run the following commands.

```
cd co-tracker
mkdir checkpoints
cd checkpoints

# download the online (multi window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth

# download the online (sliding window) model
wget https://huggingface.co/facebook/cotracker3/resolve/main/baseline_online.pth
```



### Creating the Python Environment

I will assume that you have some background in creating Python environments so will skip the initial creation. Ensure that your Python version is **V3.12**. Once you have made your environment, run the following in the root directory of your project. 

```
pip3 install -r requirements.txt
```

Please note that this will install the windows version of PyTorch. If you are on Linux, you must navigate to PyTorch's [website](https://pytorch.org/) and download the correct version for your system. Ensure you select CUDA version 12.9.



### Downloading Required Media

This is section will be completed once the project is finalized.



