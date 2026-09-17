# Designing Patient-Specific Orbital Implants with Just Your Words （In Preparation）

Code repository for **OrbitalPSI-GPT**, accompanying the work:

> **A Text-Driven Clinician-in-the-Loop Approach to Accessible Patient-Specific Orbital Implant Design in Low-Resource Settings.**


## 🎥 Demo Video

**Zenodo (Restricted Access):** https://doi.org/10.5281/zenodo.17936808
   
<details>
<summary><b>Availability</b> (click to expand)</summary>

- The video is currently under **Restricted Access**.
- It will be made **publicly available after the publication** of our work.
- The Zenodo DOI remains stable and can be cited at any time.
- Researchers wishing to access the video before publication may request access directly via the Zenodo page or email: gary.gaoyao@gmail.com.

</details>

## 🧠 System Overview
OrbitalPSI-GPT enables surgeons to design patient-specific orbital implants through natural-language instructions, without requiring CAD expertise or proprietary software, particularly in low-resource settings. For more information or technical details, please refer to our paper (to be updated upon publication):

<img src="assets/Github.svg" alt="Figure" width="1400">

## ✅ Clinician-in-the-loop Workflow （Our highlights）

We break the design process into a few simple checkpoints (CP0–CP4).  
You stay in control from start to finish — using just your words.

<details>
<summary><b>Workflow</b> (click to expand)</summary>

1. **CP0 — Case setup: Tell the system what’s going on**  
   You start by giving a short description of the case (which side, what type of defect, etc.).  
   The system automatically extracts the key information and prepares the case for you.

2. **CP1–CP2 — Anatomy check: Make sure the “map” is correct**  
   The system loads the CT scan and the segmentation results.  
   You quickly look at the 3D view to confirm everything is labelled correctly.  
   👉 **If you don’t approve it, the system will not generate any implant.**

3. **CP3 — First draft: The system creates an initial implant shape**  
   Using expert-designed templates mapped to the patient’s orbit virtual reconstructions, the system generates a smart “first draft” implant surface that roughly fits the defect.  
   You review this baseline — and only after you say **“Confirm”** in the chat does the system move on.

4. **CP4 — Chat & adjust: Refine the boundaries through simple text commands**  
   The system interprets your text and updates the 3D model in real time. You can go back and forth as many times as needed until it looks right.

</details>

## 📦 Assets & Releases (Our Open Access Promise)

<details>
<summary><b>Open Source Promise</b> (Click to expand)</summary>

**1. Standalone Software (One-stop Solution):**
*(Link to be updated)* - A portable `.exe` for instant usage. No setup required.

**2. Research Assets:**
For those interested in our AI framework or modular implementation, you can access our datasets and weights here:
* **Dataset (Automated Landmark Detection):** *(Link to be updated)*
* **Model Weights (Automated Landmark Detection):** *(Link to be updated)*

</details>


## 📚 Citation

If you use **OrbitalPSI-GPT** in your research or clinical work, please cite:

> Gao Y, *et al.*  
> **“A Text-Driven Clinician-in-the-Loop Approach to Accessible Patient-Specific Orbital Implant Design in Low-Resource Settings.”**  
> 2025. (Manuscript in preparation)

### BibTeX (update when the paper is accepted)

```bibtex
@article{gao2025orbitalpsigpt,
  title   = {A Text-Driven Clinician-in-the-Loop Approach to Accessible Patient-Specific Orbital Implant Design in Low-Resource Settings},
  author  = {Yao Gao et al.},
  year    = {2025},
  note    = {Manuscript in preparation},
}
```

### Restricted Model Access

The source code and benchmarking materials are publicly available in this repository.

The original trained model weights and fitted statistical shape model used in the study are not publicly distributed because they were developed using institutionally governed clinical data.

Researchers interested in accessing these materials should complete the [Model Access Request](MODEL_ACCESS_REQUEST.md). Requests will be reviewed subject to applicable UZ Leuven ethical and data-governance requirements.
