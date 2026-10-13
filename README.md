# ProSaTab-ACP

*Anticancer peptide prediction using SaProt and ProtT5 representations with TabPFN.*

## Overview

ProSaTab-ACP predicts whether a peptide is an anticancer peptide (ACP). The prediction pipeline combines SaProt and ProtT5 protein language model representations, a StableLasso feature-selection chain, and a TabPFN classifier.

This repository currently contains the prediction interface, prediction code, selector files, and a fitted TabPFN model.

## Installation

Python 3.10 is the reference environment.

The predictor requires PyTorch, Transformers, TabPFN, NumPy, scikit-learn, and SentencePiece. A CUDA-enabled PyTorch build can accelerate inference; CPU inference is also available.

## Prediction

Launch the graphical interface:

```bash
python app_tkinter.py
```

The fitted TabPFN model and selector files are provided under `models/`. The ProtT5 model path in `models/model_config.json` must be changed to a portable path or accompanied by setup instructions before release.

## Contact

For questions or technical issues, contact:

**Email:** zuoyun@jiangnan.edu.cn
