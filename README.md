# OOD Volcanic Classification

This repository contains the code implementation for the article:

**"Volcano Seismic Event Recognition and OOD Detection using Multi-Representation Deep Learning: Insights from Nevados del Chillán,"**  

## Overview

The code demonstrates how Out-of-Distribution (OOD) detectors can improve the discrimination of non-volcanic events in a multi-class classification task involving six seismic event types. It includes:

- A Jupyter notebook (`run_ood_detector_demo.ipynb`) illustrating the detection of OOD events for each of the six classes studied in the paper and visualizations of class activation maps (CAMs) for each example to interpret the model's decision.
- Jupyter notebooks (`model_training.ipynb` & `ood_detector.ipynb`) to replicate the training of the models and OOD detectors.


## Dataset & Pretrained Models

The full dataset used in the study, including:

- All 6,962 labeled traces  
- Weights for the four trained classification models  
- Corresponding OOD detectors for each model  

is available via Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15461242.svg)](https://doi.org/10.5281/zenodo.15461242)

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/camilo-espinosa/ood-volcanic-classification
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

To replicate the training of the models and the OOD detectors, use the notebooks:

```bash
model_training.ipynb
```
&
```bash
ood_detector.ipynb
```

## Citation

[1] C. Espinosa-Curilem, D. Basualto, M. Curilem, and F. Huenupan, “Volcano Seismic Event Recognition and OOD Detection using Multi-Representation Deep Learning: Insights from Nevados del Chillán,” Journal of Volcanology and Geothermal Research, vol. 466, p. 108406, Oct. 2025, doi: 10.1016/j.jvolgeores.2025.108406. Available: http://dx.doi.org/10.1016/j.jvolgeores.2025.108406



