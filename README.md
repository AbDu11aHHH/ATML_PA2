# ATML PA2 — Domain Adaptation & Generalization

This repository contains implementations and analyses for **EE5102/CS6302 – Advanced Topics in Machine Learning (Fall 2025) Assignment 2**.  
The overall objective is to study **Domain Adaptation (DA)** and **Domain Generalization (DG)** methods for improving robustness under distribution shifts, progressing from classical UDA techniques to modern prompt learning with CLIP.

---

## Project Overview

### Task 1 – Unsupervised Domain Adaptation (UDA)

Train a model on a labeled **source** domain and adapt it to an unlabeled **target** domain.  

**Methods implemented and compared:**

1. **Source-Only Baseline** — Standard supervised training on source data; evaluated on both domains to quantify domain shift.  
2. **Domain Alignment Methods**
   - **DAN (Deep Adaptation Network)** – Statistical alignment via MMD/kernels.  
   - **DANN (Domain-Adversarial Neural Network)** – Adversarial alignment using a gradient reversal layer.  
   - **CDAN (Conditional Adversarial Domain Adaptation)** – Class-aware feature alignment.  
   Evaluate improvements in target accuracy and any negative-transfer effects (e.g., class confusion).  
3. **Self-Training on Target (Pseudo-Labeling)** — Use source-trained model predictions as pseudo-labels for target fine-tuning.  
4. **Concept Shift & Rare-Class Scenarios** — Examine label imbalance and concept-shift robustness using t-SNE plots and confusion matrices.

> **Goal:** Understand the trade-off between domain invariance and class discriminability, and visualize how alignment affects feature overlap between domains.

---

### Task 2 – Domain Generalization (DG)

Train on **multiple labeled source domains** to generalize to an **unseen target domain** (no target data during training).

**Methods implemented:**

1. **ERM Baseline (Empirical Risk Minimization)** — Train a standard classifier on merged source domains; evaluate per-domain and unseen target accuracy.  
2. **IRM (Invariant Risk Minimization)** — Encourage representations that share the same optimal classifier across domains via gradient variance penalty.  
3. **Group DRO (Distributionally Robust Optimization)** — Optimize for the worst-performing domain to improve balance and robustness.  
4. **SAM (Sharpness-Aware Minimization)** — Train with a perturbed loss to find flatter minima and enhance out-of-domain robustness.

> **Goal:** Evaluate which techniques improve unseen-domain performance and how invariance and flatness contribute to generalization.

---

### Task 3 – Prompt Learning with CLIP (for DA/DG)

Implements **learnable prompts** for CLIP (ViT-B/32) to explore lightweight domain adaptation.

- **Domain-Agnostic + Domain-Specific Prompt Vectors**  
- **Image–Text Similarity Classification**  
- **Gradient Conflict Analysis:** Measure cosine similarity between source and target gradients to detect interference.  
- **Visualization:** Track source/target accuracy and gradient alignment trends across epochs.  
- **Optional Extension:** Add entropy minimization or gradient alignment loss for unsupervised target regularization.

> **Goal:** Investigate prompt tuning as a bridge between domain-specific adaptation and domain-invariant generalization.

---

## ⚙️ Dependencies

All libraries are available via pip or pre-installed in Google Colab.

```bash
pip install torch torchvision transformers tqdm scikit-learn pillow matplotlib
