# Two-Stream GeneSegNet (RNA-Guided Cell Segmentation)

This repository contains the official implementation of the Two-Stream GeneSegNet architecture, modifying the core feature extraction pipeline with a unidirectional cross-attention mechanism at the bottleneck layer.

## Dataset
The dataset used in this study (Mouse Hippocampus DAPI + RNA Heatmaps) is available at:
👉 **[Download from Google Drive (Dataset.zip)](https://drive.google.com/file/d/14c9Vj9YjNj7p20saZNMuDeakhZBJvUDI/view?usp=sharing)**

Please download and place the data into a folder named `Dataset/` in the root directory before running the scripts.

## Pretrained Models
Best performing model weights for both Baseline and Two-Encoder architectures:
👉 **[Download from Google Drive (Models.zip)](https://drive.google.com/file/d/1IkBnLkSsiroaVRTq2w6xxWlpjdjuwqqg/view?usp=sharing)**

To use these models, place them in the `Code/` directory or specify their paths in the test scripts.


## Installation (Docker)
We recommend using Docker for a fully reproducible environment. 

1. **Build the image:**
```bash
docker build -t twostream-genesegnet .
```

2. **Run the container (with GPU support):**
```bash
docker run --gpus all -it -p 8888:8888 -v /path/to/your/Dataset:/workspace/TwoStream-GeneSegNet/Dataset twostream-genesegnet
```
*Note: This will start a JupyterLab server at `http://localhost:8888`.*

## Running the Code
1. Training both Baseline and Two-Encoder models for 250 Epochs with Cosine Annealing Learning Rate:
```bash
cd Code
python train_wrapper.py
```

2. Generating Test Metrics (AP, Precision, Recall, F1):
```bash
python test_twostream.py
python test_baseline.py
```
