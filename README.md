
# A Cross‑modal Lag-aware Graph Contrastive Learning Framework for Safety Domain Modeling in Nuclear Power Systems

![image](https://github.com/JinlinYY/CLGC/blob/main/Abstract_graph.png)

Official implementation for “A Cross-modal Lag-aware Graph Contrastive Learning Framework for Safety-Domain Discrimination in Nuclear Power Systems”.
The framework builds a lag-aware heterogeneous sensor graph over multi-rate operation (OP) and dose (DOSE) streams, performs temporal encoding + heterogeneous message passing, and optimizes multi-granularity contrastive objectives for robust, interpretable safety-domain inference.


---

## 📦 Data


Place datasets under `./Data/` (or symlink):

```
Data/
├── NPS/         
└── TFF/          
```
The **NPS** dataset is open-access and can be found [https://www.kaggle.com/datasets/amytai/cancernet-bca](https://github.com/thu-inet/NuclearPowerPlantAccidentData/tree/main).

The **TFF** dataset is open-access and can be found [https://www.isic-archive.com/](https://ieee-dataport.org/documents/three-phase-flow-facility).

------

## 🏋️ Training

```
python train.py

```

Use `--help` to see all arguments.

------

## ✅ Evaluation

```
python eval.py
```

This exports Accuracy / Precision / Recall / F1 / AUC / MCC / Specificity / FNR, confusion matrices, PR/ROC curves, and t-SNE plots.

------

## 🛡 Robustness Evaluation

```
python utils/robust_eval_auto.py 
```

------

## ⚡ Profiling & Throughput

Evaluate compute and throughput on **real sliding-window heterogeneous graph batches** (with warm-up):

```
python utils/profile_model.py
```
