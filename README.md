# Towards Distribution-Shift Uncertainty Estimation for Inverse Problems with Generative Priors
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository builds on the official implementation of the paper [What's in a Prior? Learned Proximal Networks for Inverse Problems](https://openreview.net/pdf?id=kNPcOaqC5r) @ [ICLR 2024](https://iclr.cc/Conferences/2024) by [Zhenghan Fang](https://zhenghanfang.github.io/), [Sam Buchanan](https://sdbuchanan.com/), and [Jeremias Sulam](https://sites.google.com/view/jsulam).

We adapt their Learned Proximal Network (LPN) framework to investigate **distribution-shift uncertainty estimation** in inverse problems.  
Our experiments focus specifically on **MNIST-based sparse-view CT reconstruction**.

👉 This repository is an extension of their work; please see the original repo for the full method and additional experiments.  
👉 **All pre-trained models and results are provided in this repository, so the `evaluate_ct.ipynb` notebook file can be executed directly without retraining.**

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

### MNIST for CT

The dataset is located in `data/mayoct/mnist`.
The `only0` subset is used for in-distribution training, while `idood_valid` is used for evaluating distribution-shift performance (in- vs out-of-distribution).


---

## How to Run the Code

Code of the main functionalities of LPN is placed in the `lpn` folder.  
Code for reproducing the MNIST experiments is placed in the `exps/mayoct` folder.

### MNIST for CT Experiment

1. **Train and Compute prior:**

```bash
bash run.sh      # trains the Learned Proximal Network (LPN)
bash run_ct.sh   # performs CT reconstruction using the trained prior
```

- Trained model: `exps/mayoct/models_mnist/lpn`
- Training logs: `exps/mayoct/models_mnist`
- CT results: `exps/mayoct/results/inverse/mayoct/tomo`

2. **Visualize results**

- Evaluation and visualization available at `evaluate_ct.ipynb`. **All pre-trained models and results are provided in this repository, so the notebook can be executed directly without retraining.**


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
