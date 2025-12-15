# 🦴 nnU-Net Orbital Bone Segmentation (Fracture-Aware) for PSI Planning

An **nnU-Net–based orbital bone segmentation pipeline** designed to generate **high-fidelity 3D reconstruction models** from raw clinical CT scans, with a specific focus on **orbital fracture cases** for **patient-specific implant (PSI)** planning.

---

## ✨ Why this matters

Orbital fracture reconstruction relies on accurate delineation of **thin orbital walls** and **defect boundaries**. Many publicly available craniofacial tools (e.g., *DentalSegmentator*) are trained primarily on **healthy-control datasets**, which can limit robustness when encountering trauma-related morphology (discontinuities, comminution, and irregular defects).

This sub-work addresses that gap by building a **fracture-aware** segmentation pipeline using **nnU-Net**, trained on **real-world orbital fracture CTs** to better reflect clinical PSI planning scenarios.

---

## 🗂️ Development Cohort

| Item | Description |
|------|-------------|
| Cohort size | **50** de-identified CT scans |
| Population | Patients with **orbital fractures** acquired during routine clinical care |
| Ethics approval | **S68944** |
| Rationale | Fracture-case cohort mirrors **defect morphology + imaging characteristics** relevant to PSI planning |

---

## 🎯 Segmentation Targets (Implant-Relevant Structures)

Expert raters annotated key orbital structures used in PSI planning, including:

- **Orbital floor**
- **Medial wall**
- **Orbital rim**
- **Defect / fracture masks**

**Quality control:** annotations were reviewed by expert raters with **adjudication** to resolve disagreements and ensure consistent ground truth.

---

## ⚙️ Preprocessing (nnU-Net Reference Configuration)

Preprocessing followed the standard nnU-Net configuration to maximize reproducibility:

- Resampling to **isotropic spacing**
- **Bone-window intensity clipping**
- **Z-score normalization**
- nnU-Net automated **dataset fingerprinting** and **configuration selection**

> No additional task-specific heuristics were introduced beyond the nnU-Net framework.

---

## 🧠 Model

- **Framework:** nnU-Net (self-configuring U-Net)
- **Training regime:** supervised training on **fracture-case cohort**
- **Goal:** robust segmentation of orbital bony structures and defect boundaries under trauma conditions

---

## ⬇️ Pretrained Weights

> **Note:** Please replace the placeholder links below with your actual release URLs (GitHub Releases / Zenodo / institutional storage). I’m not able to infer your real weight address from the text provided, so I’m **not** going to fabricate a link.

### Zenodo (best for long-term archiving) TO-BE-UPDATED
- **Zenodo record:** `https://doi.org/<YOUR_DOI>`  


## 🔁 Downstream Usage

The segmentation outputs can be used directly for:

- **3D surface reconstruction**
- **Statistical shape modeling (SSM)**
- **Implant boundary extraction / editing**
- Text-driven clinician-in-the-loop **PSI design workflows** (e.g., OrbitalPSI-GPT)
---

## 📚 References

- Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH.  
  **nnU-Net: a self-configuring method for deep learning–based biomedical image segmentation.**  
  *Nature Methods*, 2021.

- *DentalSegmentator*: [public craniofacial segmentation tool trained primarily on healthy-control datasets.](https://github.com/gaudot/SlicerDentalSegmentator)

---

## 📝 Notes

- This repository summarizes the **segmentation and annotation pipeline**.
- Clinical outcome and PSI reconstruction results are reported in the associated manuscript.
- The dataset is **not publicly released** due to patient privacy constraints.
