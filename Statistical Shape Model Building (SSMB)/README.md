# Region-Aware Multiscale GPMM for Orbital Bone Reconstruction

[![Scalismo](https://img.shields.io/badge/Scalismo-0.92.0-blue.svg)](https://scalismo.org/)
[![Scala](https://img.shields.io/badge/Scala-3.3-red.svg)](https://www.scala-lang.org/)

This repository contains the official Scalismo implementation for the **Region-Aware Multiscale Gaussian Process Morphological Model (GPMM)**. 

The pipeline performs automatic mesh alignment and constructs a statistical shape model incorporating clinical anatomical priors (Orbital Floor, Medial/Lateral Walls) using spatially weighted kernels and multi-level clustering ($k=2,3,4$).

---

## 📂 Project Structure

```text
.
├── Orbital-Bone-Dataset/                  # Data Folder (See Disclaimer below)
│   ├── reference-orbital-bone.stl         # The reference template (Fixed Topology)
│   ├── subject_01.stl                     # Example subject mesh
│   ├── subject_02.stl
│   └── ...
├── AlignModels.scala                      # Step 1: Script for Rigid/GPA Alignment
├── BuildRegionAwareMultiscaleGPMM.scala   # Step 2: Main script to build the GPMM (.h5)
└── README.md                              # Project Documentation
```

---

## 🚀 Prerequisites

Java Runtime: JDK 17 or higher

Scala CLI (recommended):

https://scala-cli.virtuslab.org/install/

Scalismo: Version 0.92.0

(All dependencies are handled automatically via script directives)

## 🛠️ Usage
Step 1: Data Alignment (Optional)
Before building the statistical model, all subject meshes must be registered to the reference template to ensure vertex-wise correspondence. This step can be performed using rigid alignment or Generalized Procrustes Analysis (GPA).
```text
scala-cli run AlignModels.scala
```
Input: Raw meshes from Orbital-Bone-Dataset/.

Output: Aligned meshes (files are either updated or saved to a processed directory).

Step 2: Build Region-Aware Model
This script implements the core methodology of the paper:

Multi-level Clustering: Automatically partitions the reference mesh into regions ($k=2,3,4$).

Clinical Importance Weighting: Assigns specific weights to the Orbital Floor ($1.0$), Medial Wall ($0.7$), etc.

Kernel Construction: Combines a Global Gaussian kernel with localized, weighted kernels.

Model Approximation: Approximates the infinite GP to generate a Low-Rank Statistical Mesh Model.

```text
scala-cli run BuildRegionAwareMultiscaleGPMM.scala
```

Output: region_aware_gpmm.h5 (Standard Scalismo statistical model format).

## ⚖️ Data Availability & Disclaimer
Important Note regarding the .stl files provided in Orbital-Bone-Dataset/:

The 3D mesh files included in this repository are intended solely for reproducibility and demonstration purposes.

Privacy & Ethics: These files do not contain data from the actual clinical patients included in our study/paper.

Origin: All meshes are derived from publicly available open-source medical imaging datasets.

Preprocessing: To facilitate code execution and ensure topological consistency for the Scalismo pipeline, the raw data was segmented and post-processed using smoothing algorithms (e.g., Laplacian smoothing) to remove artifacts and optimize mesh quality.
