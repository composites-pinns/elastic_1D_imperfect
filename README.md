# `elastic_1D_imperfect`

![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![PyTorch Version](https://img.shields.io/badge/pytorch-2.2.1-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the code accompanying the following published article.

> B. Mederos, Y. Espinosa-Almeyda, O. L. Cruz-González, R. Rodríguez-Ramos, and A. C. Alvarez (2026).\
> Physics-informed neural networks for predicting homogenized effective properties of one-dimensional composites with imperfect interface.\
> *Mathematics and Mechanics of Solids*. OnlineFirst, 8 October 2026. [DOI 10.1177/10812865261486846](https://doi.org/10.1177/10812865261486846).

## Overview

This project combines the **Asymptotic Homogenization Method (AHM)** with **Physics-Informed Neural Networks (PINNs)** to study the effective elastic behavior of one-dimensional periodic composites with imperfect interfaces.

The PINN formulation uses separate neural networks in the material phases and incorporates the governing equations, boundary conditions, and interface conditions into the loss function. The numerical predictions are compared with the corresponding analytical homogenization solution.


## Installation

The code has been tested with:

- Python **3.10**
- PyTorch **2.2.1**
- CUDA **11.8**, when GPU execution is available

Create a Conda environment:

```bash
conda create --name ahm-pinns-imperfect python=3.10
conda activate ahm-pinns-imperfect
```

Install PyTorch with CUDA support:

```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

For CPU-only execution:

```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 cpuonly -c pytorch
```

Install the remaining dependencies:

```bash
pip install numpy pandas matplotlib tqdm
```

Clone the repository:

```bash
git clone https://github.com/composites-pinns/elastic_1D_imperfect.git
cd elastic_1D_imperfect
```

## Getting Started

The numerical experiments are organized into two main folders:

- `Different_vf_FixedConstrast`
- `Different_Contrast`

The `optimizer` folder contains the optimizer used during training.

## Basic Usage

Run the training and prediction scripts from the corresponding experiment folder. For example:

```bash
python Different_vf_FixedConstrast/train_different_vf_fixedContrast.py
python Different_vf_FixedConstrast/prediction_different_vf.py
```

or

```bash
python Different_Contrast/train_different_contrast_vfractions.py
python Different_Contrast/prediction_different_contrast_vfractions.py
```

Generate the comparison figures and reconstructed displacement fields with:

```bash
python "Different_Contrast/show_C_vs_vf_ N_vs_y_fordifferent_k.py"
python Different_Contrast/reconstruction_analytical_vs_pinns.py
```

Some file paths may need to be adapted before execution.

## Results

The numerical results compare the PINN predictions with the analytical homogenization solution for the local fields, the effective elastic properties, and the reconstructed heterogeneous displacement.

<p align="center">
  <img src="figures/Figure4.png" width="700" alt="Effective Young modulus comparison">
</p>

<p align="center"><em>Effective Young's modulus predicted by the PINN and obtained analytically for a volumen fraction of 0.2.</em></p>


## Citing

If you use this code, please cite the published article.

```bibtex
@article{mederos2026imperfectinterface,
  title   = {Physics-informed neural networks for predicting homogenized effective properties of one-dimensional composites with imperfect interface},
  author  = {Mederos, Boris and Espinosa-Almeyda, Yoanh and Cruz-Gonz{\'a}lez, Oscar L. and Rodr{\'i}guez-Ramos, Reinaldo and Alvarez, Amaury C.},
  journal = {Mathematics and Mechanics of Solids},
  year    = {2026},
  month   = oct,
  note    = {OnlineFirst, published online 8 October 2026},
  doi     = {10.1177/10812865261486846}
}
```

## License

This project is open source and available under the MIT License. See the [`LICENSE`](LICENSE) file for details.
