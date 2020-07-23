# Physics-Constrained Bayesian Neural Network
[Physics-Constrained Bayesian Neural Network for Fluid Flow Reconstruction with Sparse and Noisy Data](https://arxiv.org/pdf/2001.05542.pdf)

[Luning Sun](https://scholar.google.com/citations?user=Jszc1B8AAAAJ&hl=en), [Jian-Xun Wang](http://sites.nd.edu/jianxun-wang/)

PyTorch implementation of Physics-Constrained Bayesian Neural Network.

Noisy Stenotic Flow
:-----:

CFD | Mean | Standard Deviation
:-----:|:------:|:-----:
<img height="100" src="Figures/softcfdu_noise10.png?raw=true"> | <img height="100" src="Figures/backup_softuNN_mean_noise10.png?raw=true"> | <img height="100" src="Figures/softuNN_std_noise10.png?raw=true">

Noisy Junction Flow
:-----:

CFD | Mean | Standard Deviation
:-----:|:------:|:-----:
<img height="200" src="Figures/Junction_u_CFD_noise10.png?raw=true"> | <img height="200" src="Figures/Junction_u_mean_noise10.png?raw=true"> | <img height="200" src="Figures/Junction_u_std_noise10.png?raw=true">

## Dependencies
- python 3
- PyTorch 0.4 and above

## Installation
- Install PyTorch, TensorFlow and other dependencies

- Clone this repo:
```
git clone https://github.com/Jianxun-Wang/Physics-constrained-Bayesian-deep-learning.git
cd Physics-constrained-Bayesian-deep-learning
```

## Training

### Noisy Stenotic Flow 

Train a parametric DNN surrogate for pipe flow
```
cd code
python mainsolve.py
```
### Noisy Junction Flow

To be added

## Citation

If you find this repo useful for your research, please consider to cite:

```latex
@article{sun2020physics,
  title={Physics-constrained Bayesian neural network for fluid flow reconstruction with sparse and noisy data},
  author={Sun, Luning and Wang, Jian-Xun},
  journal={arXiv preprint arXiv:2001.05542},
  year={2020}
}
```
We also have a relational research of using [PINN to build surrogate modeling without using labelled data](https://github.com/Jianxun-Wang/LabelFree-DNN-Surrogate)
## Acknowledgments

Thanks for all the co-authors and Dr. [Yinhao Zhu](https://scholar.google.com/citations?user=SZmaVZMAAAAJ&hl=en) for his valuable discussion.

Code is inspired by [cnn-surrogate](https://github.com/cics-nd/cnn-surrogate)
