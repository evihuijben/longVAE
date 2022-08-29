import argparse
import os
import torch

class BaseOptions():
    """
    This class defines options used for both the VAE and the LongVAE (the generative model).
    It also implements several helper functions such as initializing and saving the options.
    
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # dataset parameters
        self.parser.add_argument('--dataset_name', type=str, default='starmen', help ='Name of the dataset: [ADNI, starmen, celeba]')
        self.parser.add_argument('--n_channels', type=int, default=1, help='Number of channels')
        self.parser.add_argument('--isize', type=int, default=64, help='Size of the image')
        self.parser.add_argument('--varying_length', action='store_true', help='True if the longitudinal dataset has a varying number of observations per subject')
        self.parser.add_argument('--n_steps', type=int, default=10, help='Number of steps for a longitudinal dataset with a fixed number of observations per subject')
        self.parser.add_argument('--missing_data_prob', type=float, default=-1, help='Probablity of randomly removing samples from a sequence, set to -1 for including all data.')
                
        # training parameters
        self.parser.add_argument('--lr', type=float, default = 0.0001, help='learning rate')
        self.parser.add_argument('--n_epochs', type=int, default = 100, help='number of epochs')
        self.parser.add_argument('--batchsize', type=int, default=100, help='Batchsize for training')

        # model parameters
        self.parser.add_argument('--beta', type=float, default=1, help='Weight for the KL divergence loss')
        
        # base parameters
        self.parser.add_argument('--dataroot', type=str, default='datasets', help='Path to the datasets (excluding the dataset name)')
        self.parser.add_argument('--savedir', type=str, default='trained_models', help='Path to where thrained models are saved')
        self.parser.add_argument('--device', type=str, default='cuda', help='Device on which the code is run')   
        self.parser.add_argument('--seed_value', type=int, default=0, help='Seed value to make the runs deterministic')
        self.parser.add_argument('--eval_phase', type=str, default='val')
        

    def initialize(self):
        """
        Initialize base options: define paths and device.

        Returns
        -------
        None.

        """
        self.opt.dataroot = os.path.join(self.opt.dataroot, self.opt.dataset_name)
        self.opt.splits = ['train', self.opt.eval_phase]
        
        # Set device
        if not torch.cuda.is_available():
            self.opt.device = 'cpu'
        self.opt.device = torch.device(self.opt.device)    
  