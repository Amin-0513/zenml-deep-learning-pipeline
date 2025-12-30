# zenml-deep-learning-pipeline


<p align="center">
  <img src="images/banner.PNG" alt="Federated MLflow Pipeline Banner" width="500" height="300">
</p>

## Overview
**zenml-deep-learning-pipeline** is an end-to-end deep learning project built using **ZenML** to automate the training of **CNN models on MRI datasets** downloaded from **Kaggle**. The pipeline provides a structured and reproducible workflow covering data ingestion, preprocessing, model training, and evaluation.

This project is designed to demonstrate how ZenML can be used for **MLOps-driven deep learning automation**, making experiments scalable, trackable, and easy to reproduce.

## Key Features
- Automated deep learning workflow using ZenML
- CNN-based model training on MRI medical imaging data
- Dataset ingestion from Kaggle
- Modular pipeline steps for easy experimentation
- Reproducible and scalable MLOps-friendly design

## Use Case
- Medical image classification using MRI scans  
- Deep learning experimentation with CNN architectures  
- Learning and implementing ZenML pipelines for automation

### First Step

[![Frontend Repository](https://img.shields.io/badge/Front%20End-Repository-blue?logo=github)](https://github.com/Amin-0513/brain-tumor-frontend)


## Result
<p align="center">
  <img src="images/result.PNG" alt="Federated MLflow Pipeline Banner" width="500" height="300">
</p>

## ZenML WorkFLow
<p align="center">
  <img src="images/banner.PNG" alt="Federated MLflow Pipeline Banner" width="500" height="300">
</p>

## Get started

```bash
# Clone the repository
git clone https://github.com/Amin-0513/zenml-deep-learning-pipeline.git

# Navigate to project directory
cd zenml-deep-learning-pipeline

# create python environment
python -m venv mlopps

# activate python environment
mlopps\Scripts\activate

# Install dependencies
pip install -r requirments.txt
zenml["server"]
zenml init 

## Start project
uvicorn piplineapi:app --host 0.0.0.0 --port 5002 --reload

```

