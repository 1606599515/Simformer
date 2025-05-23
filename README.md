# SimFormer 
This repository contains the official PyTorch implementation of the paper:

> **SimFormer: A Multiscale Transformer Framework with Learnable Clustering for Mesh-based Simulation**  
> Anonymous author(s)  
> *Under Review / CIKM, 2025*
> 
---

## 🚀 Overview

SimFormer is a transformer-based framework for mesh-based physical simulation. It introduces:

- ✅ **Learnable clustering** that adapts to the simulation dynamics  
- ✅ **Multiscale graph modeling** via hierarchical attention  
- ✅ **Cross-attention refinement** between clusters and nodes  
- ✅ **Superior accuracy and efficiency** across four benchmarks

![Framework](assets/Framework.png)

---

## 📦 Installation

Requirements:
- Python 3.8.10
- PyTorch 1.11.0+cu113
- CUDA 11.3

```bash
conda create -n simformer python=3.8.10
conda activate simformer

pip install -r requirements.txt
```

## 📂 Datasets

We evaluate SimFormer on:

- Beam
- SteeringWheel
- Elasticity
- DrivAerNet

## 🧪 Training & Evaluation

### Train and Evaluate on Beam dataset:

```bash
python Beam-main.py --num_epochs=1000
python Beam-main.py --num_epochs=0
```

