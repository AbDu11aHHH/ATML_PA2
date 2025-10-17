# ATML_PA2

This project implements prompt learning for domain adaptation using OpenAI’s CLIP model.
The goal is to adapt CLIP from a labeled source domain (e.g., “photo”) to an unlabeled target domain (e.g., “sketch”) by training learnable text prompts while keeping the CLIP backbone frozen.

## Project Overview

The notebook implements:

Learnable domain-agnostic and domain-specific prompt vectors

CLIP-based image-text similarity classification

Gradient conflict analysis between source and target losses

Visualization of source/target accuracy and gradient cosine similarity across epochs


## Dependencies
All dependencies can be installed via pip or are pre-installed in Google Colab.
```
pip install torch torchvision transformers tqdm scikit-learn pillow matplotlib

```
