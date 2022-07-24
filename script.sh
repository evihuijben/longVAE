#!/bin/bash


# to change
n_epochs=1000
eval_bs=1
starmen_full_run=BetaVAE_training_2022-07-23_11-55-51
starmen_missing_run=BetaVAE_training_2022-07-22_16-30-20
#celeba_full_run=
#celeba_missing_run=

run=BetaVAE_training_YYYY-MM-DD_HH-MM-SS

##### To reproduce starmen trainings of the paper #####

## VAE Starmen full data & missing data
n_epochs=100
#python train_VAE.py --dataset_name starmen --network_name default --latent_dim 8 --n_channels 1 --isize 64 --n_steps 10 --missing_data_prob -1 --lr 0.001 --n_epochs $n_epochs --batchsize_train 100
#python train_VAE.py --dataset_name starmen --network_name default --latent_dim 8 --n_channels 1 --isize 64 --n_steps 10 --missing_data_prob -0.5 --lr 0.001 --n_epochs $n_epochs --batchsize_train 100

## longVAE Starmen full data & missing data
#n_epochs=100
#python train_longVAE.py --run $starmen_full_run --dataset_name starmen --isize 64 --n_steps 10 --missing_data_prob -1 --lr 0.00001 --n_epochs $n_epochs --batchsize_train 100 --LongVAE_priors 0.3 --hidden 512
#python train_longVAE.py --run $starmen_missing_run  --dataset_name starmen --isize 64 --n_steps 10 --missing_data_prob -0.5 --lr 0.00001 --n_epochs $n_epochs --batchsize_train 100 --LongVAE_priors 0.3 --hidden 512





##### To reproduce the celeba trainings of the paper #####

### VAE CelebA full data & missing data
#python train_VAE.py --dataset_name celeba --network_name celeba --latent_dim 64 --n_channels 1 --isize 64 --n_steps 10 --missing_data_prob -1 --lr 0.00001 --n_epochs $n_epochs --batchsize_train 100 --beta 0.1
#python train_VAE.py --dataset_name celeba --network_name celeba --latent_dim 64 --n_channels 1 --isize 64 --n_steps 10 --missing_data_prob -0.5 --lr 0.00001 --n_epochs $n_epochs --batchsize_train 100 --beta 0.1


## longVAE CelebA full data & missing data
#python train_longVAE.py --run $celeba_full_run  --dataset_name celeba --isize 64 --n_steps 10 --missing_data_prob -1 --lr 0.0001 --n_epochs $n_epochs --batchsize_train 100 --LongVAE_priors 0.3 --hidden 256 --batchsize_VAE_eval $eval_bs
#python train_longVAE.py --run $celeba_missing_run  --dataset_name celeba --isize 64 --n_steps 10 --missing_data_prob -0.5 --lr 0.0001 --n_epochs $n_epochs --batchsize_train 100 --LongVAE_priors 0.3 --hidden 256 --batchsize_VAE_eval $eval_bs



##### To reproduce the ADNI trainings of the paper #####

#python train_VAE.py --dataset_name ADNI --varying_length --network_name adni --latent_dim 8 --n_channels 1 --isize 182 --lr 0.00001 --n_epochs 200 --batchsize_train 50
run=BetaVAE_training_2022-07-23_13-15-55
python train_longVAE.py --run $run --dataset_name ADNI --isize 182 --varying_length --lr 0.0001 --n_epochs 1000 --batchsize_train 1 --LongVAE_priors 1 --hidden 512
