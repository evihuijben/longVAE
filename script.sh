#!/bin/bash


##### To reproduce starmen trainings of the paper #####

## VAE Starmen full data & missing data
python train_VAE.py --dataset_name starmen --network_name default --latent_dim 8 --n_channels 1 --isize 64 --n_steps 10 --lr 0.001 --n_epochs 100 --batchsize_train 100 --missing_data_prob -1
python train_VAE.py --dataset_name starmen --network_name default --latent_dim 8 --n_channels 1 --isize 64 --n_steps 10 --lr 0.001 --n_epochs 100 --batchsize_train 100 --missing_data_prob -0.5

## longVAE Starmen full data & missing data (define the folders of the pretrained VAE below)
starmen_full_run=BetaVAE_training_YYYY-MM-DD_HH-MM-SS
starmen_miss_run=BetaVAE_training_YYYY-MM-DD_HH-MM-SS
python train_longVAE.py --run $starmen_full_run --dataset_name starmen --isize 64 --n_steps 10 --lr 0.0001 --n_epochs 1000 --batchsize_train 100 --LongVAE_priors 0.3 --hidden 512 --missing_data_prob -1
python train_longVAE.py --run $starmen_miss_run --dataset_name starmen --isize 64 --n_steps 10 --lr 0.0001 --n_epochs 1000 --batchsize_train 100 --LongVAE_priors 0.3 --hidden 512 --missing_data_prob -0.5





##### To reproduce the celeba trainings of the paper #####

## VAE CelebA full data & missing data
python train_VAE.py --dataset_name celeba --network_name celeba --latent_dim 64 --n_channels 1 --isize 64 --n_steps 10 --lr 0.00001 --n_epochs 100 --batchsize_train 100 --beta 0.1 --missing_data_prob -1
python train_VAE.py --dataset_name celeba --network_name celeba --latent_dim 64 --n_channels 1 --isize 64 --n_steps 10 --lr 0.00001 --n_epochs 100 --batchsize_train 100 --beta 0.1 --missing_data_prob -0.5


## longVAE CelebA full data & missing data (define the folders of the pretrained VAE below)
celeba_full_run=BetaVAE_training_YYYY-MM-DD_HH-MM-SS
celeba_miss_run=BetaVAE_training_YYYY-MM-DD_HH-MM-SS
python train_longVAE.py --run $celeba_full_run  --dataset_name celeba --isize 64 --n_steps 10 --lr 0.0001 --n_epochs 1000 --batchsize_train 100 --LongVAE_priors 0.3 --hidden 256 --batchsize_VAE_eval 100 --missing_data_prob -1
python train_longVAE.py --run $celeba_miss_run  --dataset_name celeba --isize 64 --n_steps 10 --lr 0.0001 --n_epochs 1000 --batchsize_train 100 --LongVAE_priors 0.3 --hidden 256 --batchsize_VAE_eval 100 --missing_data_prob -0.5



##### To reproduce the ADNI trainings of the paper #####

## VAE ADNI
python train_VAE.py --dataset_name ADNI --varying_length --network_name adni --latent_dim 8 --n_channels 1 --isize 182 --lr 0.00001 --n_epochs 200 --batchsize_train 50

## longVAE ADNI (define the folders of the pretrained VAE below)
adni_run=BetaVAE_training_YYYY-MM-DD_HH-MM-SS
python train_longVAE.py --run $adni_run --dataset_name ADNI --isize 182 --varying_length --lr 0.0001 --n_epochs 1000 --batchsize_train 1 --LongVAE_priors 1 --hidden 512
