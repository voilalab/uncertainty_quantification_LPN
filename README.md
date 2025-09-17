# Towards Distribution-Shift Uncertainty Estimation for Inverse Problems with Generative Priors
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository builds on the official implementation of the paper [What's in a Prior? Learned Proximal Networks for Inverse Problems](https://openreview.net/pdf?id=kNPcOaqC5r) @ [ICLR 2024](https://iclr.cc/Conferences/2024) by [Zhenghan Fang](https://zhenghanfang.github.io/), [Sam Buchanan](https://sdbuchanan.com/), and [Jeremias Sulam](https://sites.google.com/view/jsulam).

We adapt their Learned Proximal Network (LPN) framework to investigate **distribution-shift uncertainty estimation** in inverse problems.  
Our experiments focus specifically on **MNIST-based sparse-view CT reconstruction**.

👉 This repository is an extension of their work; please see the original repo for the full method and additional experiments.  
👉 Detailed documentation of our extensions will be updated soon.

---

## Installation

The code is implemented with Python 3.9.16 and PyTorch 1.12.0.  
Install the conda environment:

```bash
conda env create -f environment.yml
```

Install the `lpn` package:

```bash
pip install -e .
```

---

## Dataset Preparation

The datasets are placed in the `data/` folder.

### MNIST

The dataset is already in `data/mnist` with the following structure:

```
data/
└── mnist
    ├── labels.npy
    └── mnist.npy
```

---

## How to Run the Code

Code of the main functionalities of LPN is placed in the `lpn` folder.  
Code for reproducing the MNIST experiments is placed in the `exps/mnist` folder.

### MNIST Experiment

1. **Train and Compute prior:**

```bash
bash run.sh
```

- Model will be saved at `exps/mnist/experiments/mnist/model.pt`.
- Results will be saved in `exps/mnist/experiments/mnist/prior`.

2. **Visualize results**

- Evaluation and visualization available at `exps/mayoct/results/inverse/mayoct/tomo/evaluate_ct.ipynb`

---


## Acknowledgements

This work builds directly on:

- [What's in a Prior? Learned Proximal Networks for Inverse Problems](https://openreview.net/pdf?id=kNPcOaqC5r) (ICLR 2024)  
- [Original implementation](https://github.com/zhenghanfang/learned-proximal-networks)

Additional references:

- [scico](https://github.com/lanl/scico)  
- [Prox-PnP](https://github.com/samuro95/Prox-PnP)  
- [unrolling_meets_data_driven_regularization](https://github.com/Subhadip-1/unrolling_meets_data_driven_regularization)  
- [odl](https://odlgroup.github.io/odl/)

---
<!-- 
## References

If you find the code useful, please also cite the original authors:

```bibtex
@inproceedings{
    fang2024whats,
    title={What's in a Prior? Learned Proximal Networks for Inverse Problems},
    author={Zhenghan Fang and Sam Buchanan and Jeremias Sulam},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024}
}
``` -->
