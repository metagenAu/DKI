# DKI (Data-driven Keystone species Identification)
This is a Pytorch implementation of DKI, as described in our paper:

Wang, X.W., Sun, Z., Jia, H., Michel-Mata, S., Angulo, M.T., Dai, L., He, X., Weiss, S.T. and Liu, Y.Y. [Identifying keystone species in microbial communities using deep learning]. bioRxiv, pp.2023-03 (2023). 

<p align="center">
  <img src="Paper/DKI.png" alt="demo" width="600" height="470" style="display: block; margin: 0 auto;">
</p>


We have tested this code for Python 3.8.13 and R 4.1.2.

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [Data type for DKI](#Data-type-for-DKI)
- [How the use the DKI framework](#How-the-use-the-DKI-framework)

# Overview

Previous studies suggested that microbial communities harbor keystone species whose removal can cause a dramatic shift in microbiome structure and functioning. Yet, an efficient method to systematically identify keystone species in microbial communities is still lacking. This is mainly due to our limited knowledge of microbial dynamics and the experimental and ethical difficulties of manipulating microbial communities. Here, we propose a Data-driven Keystone species Identification (DKI) framework based on deep learning to resolve this challenge. Our key idea is to implicitly learn the assembly rules of microbial communities from a particular habitat by training a deep learning model using microbiome samples collected from this habitat. The well-trained deep learning model enables us to quantify the community-specific keystoneness of each species in any microbiome sample from this habitat by conducting a thought experiment on species removal. We systematically validated this DKI framework using synthetic data generated from a classical population dynamics model in community ecology. We then applied DKI to analyze human gut, oral microbiome, soil, and coral microbiome data. We found that those taxa with high median keystoneness across different communities display strong community specificity, and many of them have been reported as keystone taxa in literature. The presented DKI framework demonstrates the power of machine learning in tackling a fundamental problem in community ecology, paving the way for the data-driven management of complex microbial communities.


# Refactor in progress

The original code is preserved at `legacy/DKI_original.py` and
`Keystoneness_computing.R`. A staged refactor is underway in the
`dki/` Python package — each phase ships behind its own sanity check
before the next is started.

| Phase | Status | What it adds |
|---|---|---|
| **1. Faithful refactor + batched integration** | ✅ landed | `dki/` package (`data`, `model`, `losses`, `infer`, `train`, `keystoneness`), batched `dopri5` replicator ODE (`rtol=1e-5`, `atol=1e-7`, `t=[0,100]`), cosine LR, gradient clipping at 1.0, early stop on val BC, best-val checkpoint, auto CUDA/MPS/CPU. CLI: `python -m dki.train --data data`. ~600× per-epoch speedup over the original. |
| **2. Nonlinear ODEFunc + composite loss** | ✅ landed | `--nonlinear` makes per-capita fitness `fc2(SiLU(fc1(y)))` with hidden dim `hidden_mult·N` (default `2N`). The original cNODE2's two stacked `Linear` layers without activation are provably equivalent to a single `Linear` (W2·W1 = W); `tests/test_phase2.py` exhibits a near-zero best-affine-fit residual for the legacy collapse and a large one for the SiLU version. `--loss composite --alpha 0.3` adds `α·BC + (1−α)·CLR-MSE` to repair rare-species accuracy. |
| **3. Deep-equilibrium reformulation** | ✅ landed | `--mode deq` (`dki/deq.py`): solves the replicator fixed point with **safeguarded** Anderson acceleration (50 iters, tol 1e-6) on a simplex-preserving mirror-descent map, backprop via the implicit function theorem (adjoint linear solve in a backward hook). Per-row safeguarding rejects any extrapolation that raises the residual, so it lands on the *stable* equilibrium the ODE flow reaches; falls back to the ODE solver on non-convergence. Matches the ODE within BC < 0.02 on the potential-game regression test. |
| **4. Ensembles + uncertainty + null-model normalisation** | ✅ landed | `dki/ensemble.py`: `train_ensemble` fits K bootstrap-resampled models (default K=5, distinct seeds), `EnsemblePredictor.predict` returns `(mean, std)`. Keystoneness module gains an **alternative** z-score calibration (`null_model_keystoneness`, up to 50 abundance-matched null species per (sample, species)) alongside — not replacing — the classical `(1−p)` formula. |
| **5. Shapley keystoneness** *(extension)* | ✅ landed | `dki/extensions/shapley.py`: Monte-Carlo Shapley (default `n_perm=200`) for a **different question** — synergy/redundancy-aware contribution — not a fix to the classical definition. Value function `v(S)=1−BC(q_Ω, q_S)`; reported as `k_shapley_synergistic`, never replacing `k_classical` or `k_zscore`. |
| **6. Self-consistency regulariser** | ✅ landed | `--consistency-weight 0.1` adds a training-time auxiliary loss (`self_consistency_loss`): mask one present species, predict `q'`, re-feed `q'` as an initial condition, require the re-integrated prediction matches `q'` under BC. Pushes predictions to be genuine fixed points (tighter ensemble std for keystoneness). |

Design notes pinned by the project:

* The metacommunity assumption (same `f` across all samples; only `z`
  varies) is preserved — the ODEFunc takes **only `y`**. No covariate
  conditioning, no hypernetworks, no context-dependent interactions.
* Classical Paine-style keystoneness stays in the default output; the
  null z-score and Shapley results land as alternatives, not
  replacements.

A Colab notebook that runs the whole pipeline end-to-end lives at
[`notebooks/dki_colab.ipynb`](notebooks/dki_colab.ipynb)
([open in Colab](https://colab.research.google.com/github/metagenAu/DKI/blob/claude/peaceful-goodall-4AC8l/notebooks/dki_colab.ipynb)).

## Why the nonlinearity matters (Phase 2)

SI §2 of the paper claims that going from cNODE (one `Linear`) to cNODE2
(two stacked `Linear` layers) captures "non-linear interactions between
species." Mathematically, `W₂(W₁ y) = (W₂ W₁) y` is still a single
linear map — the two-Linear-without-activation construction is
equivalent in expressivity to a single `Linear` layer with `W = W₂ W₁`.
The overall dynamics are nonlinear only because of the replicator
wrapping, and that was already true in cNODE. Phase 2 inserts a SiLU
between the layers so the *fitness* function itself becomes nonlinear,
and ships a regression test that exhibits a 1000-input residual ≈ 0
for the legacy product collapse and ≫ 0 for the SiLU version.

# Repo Contents
(1) A synthetic dataset to test the Data-driven Keystone species Identification (DKI) framework.

(2) Python code to predict the species composition using species assemblage (cNODE2) and R code to compute keystoneness.

(3) Predicted species composition after removing each present species in each sample.

# Data type for DKI
## (1) Ptrain.csv: matrix of taxanomic profile of size N*M, where N is the number of taxa and M is the sample size (without header).

|           | sample 1 | sample 2 | sample 3 | sample 4 |
|-----------|----------|----------|----------|----------|
| species 1 | 0.45     | 0.35     | 0.86     | 0.77     |
| species 2 | 0.51     | 0        | 0        | 0        |
| species 3 | 0        | 0.25     | 0        | 0        |
| species 4 | 0        | 0        | 0.07     | 0        |
| species 5 | 0        | 0        | 0        | 0.17     |
| species 6 | 0.04     | 0.4      | 0.07     | 0.06     |

## (2) Thought experiment: thought experiemt was realized by removing each present species in each sample. This will generated three data type.

* Ztest.csv: matrix of perturbed species collection of size N*C, where N is the number of taxa and C is the total perturbed samples (without header).

|           | sample 1 | sample 2 | sample 3 | sample 4 | sample 5 | sample 6 | sample 7 | sample 8 | sample 9 | sample 10 | sample 11 | sample 12 |
|-----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|-----------|-----------|
| species 1 | 0        | 1        | 1        | 0        | 1        | 1        | 0        | 1        | 1        | 0         | 1         | 1         |
| species 2 | 1        | 0        | 1        | 0        | 0        | 0        | 0        | 0        | 0        | 0         | 0         | 0         |
| species 3 | 0        | 0        | 0        | 1        | 0        | 1        | 0        | 0        | 0        | 0         | 0         | 0         |
| species 4 | 0        | 0        | 0        | 0        | 0        | 0        | 1        | 0        | 1        | 0         | 0         | 0         |
| species 5 | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 1         | 0         | 1         |
| species 6 | 1        | 1        | 0        | 1        | 1        | 0        | 1        | 1        | 0        | 1         | 1         | 0         |

* Species_id: a list indicating which species has been removed in each sample.

| species |
|---------|
| 1       |
| 2       |
| 6       |
| 1       |
| 3       |
| 6       |
| 1       |
| 4       |
| 6       |
| 1       |
| 5       |
| 6       |

* Sample_id: a list indicating which sample that the species been removed.

| sample |
|--------|
| 1      |
| 1      |
| 1      |
| 2      |
| 2      |
| 2      |
| 3      |
| 3      |
| 3      |
| 4      |
| 4      |
| 4      |

# How to use the DKI framework
## Step 1: Predict species compostion using perturbed species assemblage
Run Python code "DKI.py" by taking Ptrain.csv and Ztest.csv as input will output the predicted microbiome composition using perturbed species colloction matrix Ztest.csv.
The output file qtst.csv:

|           | sample 1  | sample 2  | sample 3  | sample 4   | sample 5   | sample 6   | sample 7   | sample 8  | sample 9  | sample 10  | sample 11  | sample 12 |
|-----------|-----------|-----------|-----------|------------|------------|------------|------------|-----------|-----------|------------|------------|-----------|
| species 1 | 0.0000000 | 0.000000  | 0.0000000 | 0.92458308 | 0.92458308 | 0.92458308 | 0.9245831  | 0.4725695 | 0.4729691 | 0.91488211 | 0.8053058  | 0.8053058 |
| species 2 | 0.8315174 | 0.0000000 | 0.000000  | 0.0000000  | 0.00000000 | 0.00000000 | 0.00000000 | 0.0000000 | 0.5274305 | 0.0000000  | 0.00000000 | 0.0000000 |
| species 3 | 0.0000000 | 0.8287832 | 0.000000  | 0.0000000  | 0.00000000 | 0.00000000 | 0.00000000 | 0.0000000 | 0.0000000 | 0.5270309  | 0.00000000 | 0.0000000 |
| species 4 | 0.0000000 | 0.0000000 | 0.212941  | 0.0000000  | 0.00000000 | 0.00000000 | 0.00000000 | 0.0000000 | 0.0000000 | 0.0000000  | 0.08511789 | 0.0000000 |
| species 5 | 0.0000000 | 0.0000000 | 0.000000  | 0.4444696  | 0.00000000 | 0.00000000 | 0.00000000 | 0.0000000 | 0.0000000 | 0.0000000  | 0.00000000 | 0.1946942 |
| species 6 | 0.1684826 | 0.1712168 | 0.787059  | 0.5555304  | 0.07541692 | 0.07541692 | 0.07541692 | 0.0754169 | 0.0000000 | 0.0000000  | 0.00000000 | 0.0000000 |

## Step 2: Compute the keystoneness
Run R code Keystoneness_computing.R to compute the keystonenss of each present in each sample. The output file:

| keystoneness | sample | species |
|--------------|--------|---------|
| 5.576585e-02 | 1      | 1       |
| 5.680769e-02 | 2      | 1       |
| 4.133107e-02 | 3      | 1       |
| 6.768209e-02 | 4      | 1       |
| 3.948267e-05 | 1      | 2       |
| 4.027457e-05 | 2      | 3       |
| 7.398025e-05 | 3      | 4       |
| 5.262661e-05 | 4      | 5       |
| 4.576021e-03 | 1      | 6       |
| 3.072820e-03 | 2      | 6       |
| 7.672017e-03 | 3      | 6       |
| 1.067806e-02 | 4      | 6       |

Each row represent the keystonenes of a species in a particular sample.


