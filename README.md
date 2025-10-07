# IrisFormer (VIS Adaptation): Transformer for Visible Spectrum Iris Recognition

This repository extends the original [IrisFormer](https://github.com/XianyunSun/IrisFormer) framework to the **visible spectrum (VIS)** iris recognition setting.  
It introduces modifications and training protocols specifically aimed at addressing the challenges of **illumination variability**, **pigmentation**, and **blur** commonly encountered in VIS iris imaging.  

Our trained VIS model achieves **state-of-the-art performance** across standard VIS iris benchmarks (UBIRIS.v1, UBIRIS.v2, MICHE, and CUVIRIS) and is released for reproducible research.

---

## 🔍 Overview

Unlike traditional handcrafted iris recognition systems (e.g., OSIRIS), which encode phase responses into binary iris codes, **IrisFormer** leverages transformer-based attention to learn **patch-wise embeddings** directly from normalized iris strips.  
Our VIS adaptation retains the original model’s transformer backbone but integrates training and augmentation strategies tailored to the visible-light domain.

### Key Design Highlights
- **2D Relative Positional Encoding (RoPE)** — handles horizontal misalignments caused by residual eye rotation after normalization.  
- **Horizontal Pixel-Shift Augmentation** — simulates small rotational offsets between captures.  
- **Random Token Masking** — increases robustness to local occlusions and specular reflections.  
- **Patch-wise Sequential Matching** — preserves fine-grained texture and maintains local similarity signals.

---

## 🧠 Input Representation

- **Input:** Normalized grayscale iris image, size `64 × 512`.  
- The image is tokenized into non-overlapping patches, linearly projected to a fixed embedding dimension, and passed through a transformer encoder stack.  
- Final features are extracted as the **full sequence of patch embeddings**, matched via cosine similarity in a patch-wise sequential fashion.

---

## ⚙️ Training Setup

| Setting | Value |
|----------|-------|
| Dataset | UBIRIS.v2 (Visible Spectrum) |
| Pretraining | ImageNet-1k |
| Optimizer | AdamW (weight decay 0.05) |
| Learning Rate | 1 × 10⁻⁴ with cosine annealing |
| Batch Size | 32 |
| Epochs | 100 |
| Loss | Margin-based Triplet Loss |
| Augmentation | Horizontal pixel-shift + Random token masking |

All other architectural details follow the original [IrisFormer paper](https://github.com/XianyunSun/IrisFormer).

---

## 🧪 Evaluation Protocol

The evaluation setup mirrors the OSIRIS protocol for fairness.  
We train on **UBIRIS.v2** and perform **cross-dataset verification** on:
- **UBIRIS.v1**
- **MICHE**
- **CUVIRIS**

### Reported Metrics
- Biometric: FAR, FRR, EER, DET  
- Statistical: GMean, GSTD, IMean, ISTD, *d′*, AUC, ZeroFMR, ZeroFNMR  

These collectively assess both separability and score distribution stability under VIS conditions.

---

## 💾 Pretrained Model

A trained **Visible IrisFormer (VIS)** model is available for download:

📦 **[Download Trained Model (Google Drive)](https://drive.google.com/file/d/1kEOfbw7DEGKVuzRQr7dDhQVXAjqudxVd/view?usp=sharing)**

Place the downloaded weights under:
```
./checkpoint/VIS-IrisFormer/
```

---

## 🚀 Usage

### Environment Setup
```bash
pip install torch torchvision pillow scikit-learn pyeer numpy pandas wandb
```

### Training
```bash
python train.py     --position_embedding rope2d     --ft_pool map     --shift_pixel 14     --shift_posibility 0.5     --mask_ratio 0.75     --test_while_train     --save_name VIS_IrisFormer
```

### Testing
```bash
python test.py     --position_embedding rope2d     --ft_pool map     --save_report     --run_name VIS_IrisFormer
```

Results and evaluation reports will be saved in the `eval/` directory.

---

## 🧾 Citation

If you use this work, please cite our upcoming paper:

```
@article{2025VISIrisFormer,
  title={IrisFormer (VIS Adaptation): Transformer for Visible Spectrum Iris Recognition},
  author={YourName, et al.},
  journal={To appear},
  year={2025},
  note={Preprint available soon}
}
```

A link to the final paper will be updated here once published.

---

## 🙏 Acknowledgements

This implementation builds upon the original [IrisFormer](https://github.com/XianyunSun/IrisFormer) by **Xianyun Sun et al. (IEEE SPL, 2024)**.  
We extend the framework for **Visible Spectrum Iris Recognition** and release our trained models for research reproducibility.
