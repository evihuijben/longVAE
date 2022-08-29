#!/bin/bash

##### To reproduce starmen trainings of the paper #####

# VAE Starmen full data
python train_VAE.py --dataset_name starmen --network_name default --n_epochs 100 --latent_dim 8 --lr 1e-3 --beta 1 --missing_data_prob -1

## VAE starmen missing data
python train_VAE.py --dataset_name starmen --network_name default --n_epochs 100 --latent_dim 8 --lr 1e-3 --beta 1 --missing_data_prob 0.5

## LongVAE Starmen full data (define the folder of the pretrained VAE below)
starmen_full_run=BetaVAE_training_YYYY-MM-DD_HH-MM-SS
python train_longVAE.py --dataset_name starmen --n_epochs 1000 --lr 1e-4 --beta 1 --hidden 512 --run $starmen_full_run --missing_data_prob -1

## LongVAE Starmen missing data (define the folder of the pretrained VAE below)
starmen_missing_run=BetaVAE_training_YYYY-MM-DD_HH-MM-SS
python train_longVAE.py --dataset_name starmen --n_epochs 1000 --lr 1e-4 --beta 1 --hidden 512 --run $starmen_missing_run --missing_data_prob 0.5


##### To reproduce the celeba trainings of the paper #####

## VAE CelebA full data
python train_VAE.py --dataset_name celeba --network_name celeba --n_epochs 100 --latent_dim 64 --lr 1e-5 --beta 0.1 --missing_data_prob -1

## VAE CelebA missing data
python train_VAE.py --dataset_name celeba --network_name celeba --n_epochs 100 --latent_dim 64 --lr 1e-5 --beta 0.1 --missing_data_prob 0.5

## LongVAE CelebA full data (define the folder of the pretrained VAE below)
celeba_full_run=BetaVAE_training_YYYY-MM-DD_HH-MM-SS
python train_longVAE.py --dataset_name celeba --n_epochs 1000 --lr 1e-4 --beta 1 --hidden 256 --run $celeba_full_run --missing_data_prob -1

## LongVAE CelebA missing data (define the folder of the pretrained VAE below)
celeba_missing_run=BetaVAE_training_YYYY-MM-DD_HH-MM-SS
python train_longVAE.py --dataset_name celeba --n_epochs 1000 --lr 1e-4 --beta 1 --hidden 256 --run $celeba_missing_run --missing_data_prob 0.5


##### To reproduce the ADNI trainings of the paper #####

## VAE ADNI
python train_VAE.py --n_epochs 200 --lr 1e-05 --network_name adni --beta 1 --varying_length --dataset_name ADNI --isize 182

## longVAE ADNI (define the folders of the pretrained VAE below)
adni_run=BetaVAE_training_YYYY-MM-DD_HH-MM-SS
python train_longVAE.py --dataset_name ADNI --isize 182 --varying_length --batchsize 1 --n_epochs 1000 --lr 1e-4 --beta 1 --LongVAE_priors 1 --hidden 512 --run $adni_run
