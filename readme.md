# longVAE: An Image Feature Mapping Model for Continuous Longitudinal Data Completion and Generation of Synthetic Patient Trajectories

This repository contains the code corresponding with the following paper:

> Chadebec, C.<sup>\*</sup>, Huijben, E.M.C.<sup>\*</sup>, Pluim, J.P.W., Allassonnière, S.<sup>†</sup>, and Van Eijnatten M.A.J.M.<sup>†</sup>.<br> "An Image Feature Mapping Model for Continuous Longitudinal Data Completion and Generation of Synthetic Patient Trajectories". In DGM4MICCAI 2022. Held in Conjunction with MICCAI 2022, Singapore, September 22, 2022.

> **Abstract:** *Longitudinal medical image data are becoming increasingly important for monitoring patient progression. However, such datasets are often small, incomplete, or have inconsistencies between observations. Thus, we propose a generative model that not only produces continuous trajectories of fully synthetic patient images, but also imputes missing data in existing trajectories, by estimating realistic progression over time. Our generative model is trained directly on features extracted from images and maps these into a linear trajectory in a Euclidean space defined with velocity, delay, and spatial parameters that are learned directly from the data. We evaluated our method on synthetic toy data and face images, both showing simulated trajectories mimicking progression in longitudinal data. Furthermore, we applied the proposed model on a complex neuroimaging database extracted from ADNI. All datasets show that the model is able to learn overall (disease) progression over time.*

# Setup
Install the pythae libraby by following the instructions provided at [github.com/clementchadebec/benchmark_VAE](https://github.com/clementchadebec/benchmark_VAE).

# Dataset prepration
Dataset details of the used datasets (Starmen, CelebA, and ADNI) are provided in the readme files of the corresponding folders in 'datasets'. To use a custom dataset save a 'train.pt', 'val.pt', and 'test.pt' file containing a dictionary with under 'data' a torch tensor. When your custom dataset has a fixed number of observations per subject, this data tensor is saved in the format \[N\_subjects, N\_observations, image\_size, image\_size\]. When your custom dataset has a varying number of observations the dictionary with the key 'data' should contain a list with all the subjects, saved in the format \[N\_observations, image\_size, image\_size\].

# Usage

# Citing


