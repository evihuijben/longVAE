# longVAE: An Image Feature Mapping Model for Continuous Longitudinal Data Completion and Generation of Synthetic Patient Trajectories

This repository contains the code corresponding with the following paper:

> [Chadebec, C.<sup>\*</sup>, Huijben, E.M.C.<sup>\*</sup>, Pluim, J.P.W., Allassonnière, S.<sup>†</sup>, van Eijnatten, M.A.J.M.<sup>†</sup> An Image Feature Mapping Model for Continuous Longitudinal Data Completion and Generation of Synthetic Patient Trajectories. In: Deep Generative Model (DGM4MICCAI) 2022. Springer, Cham.](https://link.springer.com/chapter/10.1007/978-3-031-18576-2_6)

> **Abstract:** *Longitudinal medical image data are becoming increasingly important for monitoring patient progression. However, such datasets are often small, incomplete, or have inconsistencies between observations. Thus, we propose a generative model that not only produces continuous trajectories of fully synthetic patient images, but also imputes missing data in existing trajectories, by estimating realistic progression over time. Our generative model is trained directly on features extracted from images and maps these into a linear trajectory in a Euclidean space defined with velocity, delay, and spatial parameters that are learned directly from the data. We evaluated our method on toy data and face images, both showing simulated trajectories mimicking progression in longitudinal data. Furthermore, we applied the proposed model on a complex neuroimaging database extracted from ADNI. All datasets show that the model is able to learn overall (disease) progression over time.*

# Setup
This code is developped on a Linux system, it is not guaranteed that this code will work on another operating system. First download a compatible version of pytorch and subsequently install the pythae library by following the instructions provided at [github.com/clementchadebec/benchmark_VAE](https://github.com/clementchadebec/benchmark_VAE) and clone the longVAE repository.

# Dataset prepration
Dataset details of the used datasets (Starmen, CelebA, and ADNI) are provided in the readme files of the corresponding folders in 'datasets'. To use a custom dataset save a 'train.pt', 'val.pt', and 'test.pt' file containing a dictionary with under 'data' a torch tensor. When your custom dataset has a fixed number of observations per subject, this data tensor is saved in the format \[N\_subjects, N\_observations, image\_size, image\_size\]. When your custom dataset has a varying number of observations, the dictionary with the key 'data' should contain a list with all the subjects, saved in the format \[N\_observations, image\_size, image\_size\] and under the key 'times', there should be a list with for all the subjects the observation times saved in the format \[N\_observations, \].

# Usage
## Step 1: Train the VAE
To pretrain the VAE, run the 'train_VAE.py' file with the desired parameters. The paramters for reproducing the experiments from the paper can be seen in 'script.sh'


## Step 2: Train the genrative model (longVAE)
To train the generative model, run the 'train_longVAE.py' file with the desired parameters. The paramters for reproducing the experiments from the paper can be seen in 'script.sh'

# Citing
```bibtex
@InProceedings{ChadebecHuijben2022,
author="Chadebec, Cl{\'e}ment
and Huijben, Evi M. C.
and Pluim, Josien P. W.
and Allassonni{\`e}re, St{\'e}phanie
and van Eijnatten, Maureen A. J. M.",
title="An Image Feature Mapping Model for Continuous Longitudinal Data Completion and Generation of Synthetic Patient Trajectories",
booktitle="Deep Generative Models (DGM4MICCAI)",
year="2022",
publisher="Springer",
address="Cham",
pages="55--64",
}

```
