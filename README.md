# A Cross‑modal Lag-aware Graph Contrastive Learning Framework for Safety Domain Discrimination in Nuclear Power Systems

![image](https://github.com/JinlinYY/CLGC/blob/main/Abstract_graph.png)

Official implementation for “A Cross-modal Lag-aware Graph Contrastive Learning Framework for Safety-Domain Discrimination in Nuclear Power Systems”.
The framework builds a lag-aware heterogeneous sensor graph over multi-rate operation (OP) and dose (DOSE) streams, performs temporal encoding + heterogeneous message passing, and optimizes multi-granularity contrastive objectives for robust, interpretable safety-domain inference.

## Data

The **NPS** dataset is open-access and can be found [https://www.kaggle.com/datasets/amytai/cancernet-bca](https://github.com/thu-inet/NuclearPowerPlantAccidentData/tree/main).

The **TFF** dataset is open-access and can be found [https://www.isic-archive.com/](https://ieee-dataport.org/documents/three-phase-flow-facility).


🚀 Setup

Tested with Python ≥ 3.9 and PyTorch ≥ 2.0.
Depending on your environment, either PyTorch Geometric or DGL may be required—install the one your implementation uses.
Typical Python dependencies:
