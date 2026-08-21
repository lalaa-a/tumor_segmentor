# Brain Tumor Segmentation with 2D U‑Net (BraTS)

This repository implements a **2D U‑Net** for brain tumor segmentation from multi‑modal MRI scans, using the [BraTS 2020](https://www.med.upenn.edu/brats/) dataset. The model is trained on axial slices extracted from four modalities (FLAIR, T1, T1ce, T2) and predicts four classes: background, necrotic core, edema, and enhancing tumor.

The code is designed to handle the full 3D volumes efficiently via a custom Keras `Sequence` generator that streams slices on‑the‑fly, so it works even with limited GPU memory.

---

## Features

- **Modular data pipeline** – Loads NIfTI files, resizes slices, normalizes intensities, and remaps BraTS labels (0,1,2,4 → 0,1,2,3).
- **2D U‑Net architecture** – Configurable number of filters and dropout.
- **Multi‑modal support** – Choose any combination of modalities as input channels (e.g., `flair,t1ce` or all four).
- **Training utilities** – Combines categorical cross‑entropy with Dice loss, monitors per‑class Dice scores, and uses callbacks for checkpointing, learning rate reduction, and early stopping.
- **Evaluation** – Computes metrics on a held‑out validation split.
- **Inference** – Visualise predictions on a single slice or segment an entire 3D volume and save as NIfTI.
- **Streaming data generator** – No need to load all volumes into RAM; data is loaded patient‑by‑patient.

---

## Requirements

Install the required Python packages:

```bash
pip install tensorflow nibabel scikit-image scikit-learn matplotlib