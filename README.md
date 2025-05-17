# OOD Volcanic Classification

This repository contains the code implementation for the article:

**"Improved Non-Volcanic Event Detection at Nevados del Chillán Volcanic Complex Using Combined Domain Representations in a Single Input"**  
*(Currently under review)*

## Overview

The code demonstrates how Out-of-Distribution (OOD) detectors can improve the discrimination of non-volcanic events in a multi-class classification task involving six seismic event types. It includes:

- A Jupyter notebook (`run_ood_detector_demo.ipynb`) illustrating the detection of OOD events for each of the six classes studied in the paper.
- Visualizations of class activation maps (CAMs) for each example to interpret the model's decision.
- A folder containing six example input traces.

## Dataset & Pretrained Models

The full dataset used in the study, including:

- All 6,962 labeled traces  
- Weights for the four trained classification models  
- Corresponding OOD detectors for each model  

is available via Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15453840.svg)](https://doi.org/10.5281/zenodo.15453840)

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/ood-volcanic-classification.git
cd ood-volcanic-classification
pip install -r requirements.txt
```

**Note:** You must install a compatible version of **PyTorch with CUDA support** to enable GPU acceleration. Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions based on your system and CUDA version.

## Usage

Open and run the notebook:
```bash
run_ood_detector_demo.ipynb
```
You can test how each OOD detector responds to individual examples, and use CAM visualizations to understand which parts of the input were most relevant to each model's decision.

## Citation

Pending


