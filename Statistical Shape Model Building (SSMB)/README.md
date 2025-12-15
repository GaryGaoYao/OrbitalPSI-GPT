# Region-Aware Multiscale GPMM for Orbital Bone Reconstruction

[![Scalismo](https://img.shields.io/badge/Scalismo-0.92.0-blue.svg)](https://scalismo.org/)
[![Scala](https://img.shields.io/badge/Scala-3.3-red.svg)](https://www.scala-lang.org/)

This repository provides the official **Scalismo-based implementation** of a  
**Region-Aware Multiscale Gaussian Process Morphological Model (GPMM)** for anatomically faithful orbital bone reconstruction.

The pipeline constructs a statistical shape model by explicitly encoding **clinical anatomical priors** (orbital floor, medial wall, lateral wall, and superior wall). It combines **soft regional masks**, **multi-level clustering** ($k = 2, 3, 4$), and **spatially weighted multiscale Gaussian kernels** to model region-dependent shape variability.

---

## 📂 Project Structure

```text
.
├── Orbital-Bone-Dataset/                  # Data folder (see disclaimer below)
│   ├── reference-orbital-bone.stl         # Reference template (fixed topology)
│   ├── subject_01.stl                     # Example subject mesh (registered)
│   ├── subject_02.stl
│   └── ...
├── AlignModels.scala                      # Step 1: Rigid / GPA alignment (optional)
├── BuildRegionAwareMultiscaleGPMM.scala   # Step 2: Build the region-aware GPMM (.h5)
└── README.md                              # Project documentation
````

---

## 🚀 Prerequisites

* **Java Runtime**: JDK 17 or higher
* **Scala CLI** (recommended):
  [https://scala-cli.virtuslab.org/install/](https://scala-cli.virtuslab.org/install/)
* **Scalismo**: Version **0.92.0**

> All dependencies are handled automatically via the script directives in the Scala files.

---

## 🛠️ Usage

### Step 1: Data Alignment (Optional)

Before building the statistical model, all subject meshes must be registered to the reference template to ensure **vertex-wise correspondence**. This step can be performed using rigid alignment or **Generalized Procrustes Analysis (GPA)**.

```bash
scala-cli run AlignModels.scala
```

* **Input**: Raw meshes in `Orbital-Bone-Dataset/`
* **Output**: Aligned meshes (either overwritten or saved to a processed directory)

> ⚠️ If your meshes have already been registered using an external pipeline (e.g., ALD), this step can be skipped.

---

### Step 2: Build the Region-Aware Multiscale GPMM

This script implements the core methodology described in the paper:

* **Multi-level clustering**
  The reference template is partitioned into soft regions using $k = 2, 3, 4$.

* **Clinical importance weighting**
  Region-specific weights are assigned based on clinical relevance:

  * Orbital floor: **1.0**
  * Medial wall: **0.7**
  * Lateral wall: **0.4**
  * Superior wall: **0.2**

* **Kernel construction**
  A region-aware multiscale Gaussian kernel is constructed by combining:

  * A global low-frequency kernel
  * Localized, spatially weighted radial basis function (RBF) kernels

* **Low-rank model approximation**
  The infinite-dimensional Gaussian process is approximated using **Pivoted Cholesky decomposition**, yielding a tractable low-rank GPMM.

```bash
scala-cli run BuildRegionAwareMultiscaleGPMM.scala
```

**Output:**

```text
region_aware_gpmm.h5
```

(Standard Scalismo statistical model format)

---

## ⚖️ Data Availability & Disclaimer

### Important Note on `Orbital-Bone-Dataset/`

The 3D mesh files provided in this repository are intended **solely for reproducibility and demonstration purposes**.

* **Privacy & Ethics**
  The meshes do **not** originate from actual clinical patients included in the study.

* **Data Origin**
  All meshes are derived from publicly available **open-source medical imaging datasets**.

* **Preprocessing**
  To ensure compatibility with the Scalismo pipeline and consistent topology, the raw data were segmented and post-processed using **mesh smoothing techniques** (e.g., Laplacian smoothing) to remove artifacts and improve mesh quality.

---

## 📚 Citation

If you use this code or methodology in your research, please cite: 

https://scalismo.org/
---

## 📄 License

This project is released under an open-source license.
Please specify the license (e.g., MIT) before public release.
