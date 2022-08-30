from .base_options import BaseOptions
import os
import json
import datetime
from utils import save_options

class LongVAEOptions(BaseOptions):
    """
    This class includes the options for training the LongVAE (the generative
    model, step 2). It also includews shared options defined in BaseOptions.
    """
    
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--run', type=str, required=True, help='Name of the pretrained VAE run: BetaVAE_training_YYYY-MM-DD_HH-MM-SS')
        self.parser.add_argument('--LongVAE_priors', type=float, default=0.3, help='Prior variance for KL divergence of velocity (eta) and delay (tau)')
        self.parser.add_argument('--hidden', type=int, default=512, help='Number of featues in hidden layer of RNN')
        self.parser.add_argument('--save_freq', type=int, default=1000, help='Frequency for intermediate saving of model weights')
        
        self.parser.add_argument('--batchsize_VAE_eval', type=int, default=None, help='Batchsize for inference of pretrained VAE')
        
        # Parameters for synthesis
        self.parser.add_argument('--no_train', action='store_true', help="Set to true if a pretrained model should be loaded for inference")
        self.parser.add_argument('--pretrained_longVAE_ID', type=str, help="ID of a pretrained model which can be used for generation, imputation and synthesis")
        self.parser.add_argument('--longVAE_load_epoch', type=int, help="Which saved epoch is used for inference. If no is provided, the best epoch on validation set is used.")
        self.parser.add_argument('--eval_times', type=str, help="At what times times the mapping function should be evaluated when using varying lenght sequences")
        
        
    def parse(self):
        """
        Parse arguments, initialize them, and load predefined parameters
        defined by the pretrained VAE.

        Returns
        -------
        opt : argparser
            Parameters defining this run.

        """
        self.opt = self.parser.parse_args()
        self.initialize()        
        
        # initialize specific arguments for longVAE based on the pretrained VAE
        self.opt.trained_VAE_path = os.path.join(self.opt.savedir, 
                                                 'VAE', 
                                                 self.opt.dataset_name, 
                                                 self.opt.run, 
                                                 'final_model')
        with open(os.path.join(self.opt.trained_VAE_path, 'model_config.json')) as inf:
            self.opt.VAE_config = json.load(inf)
        self.opt.latent_dim = self.opt.VAE_config['latent_dim']
        self.opt.input_dim = self.opt.VAE_config['input_dim']
        
        # Define new time stamp for longVAE training
        if self.opt.no_train:
            self.opt.splits = [self.opt.eval_phase]
            self.opt.ID = self.opt.pretrained_longVAE_ID
            if self.opt.longVAE_load_epoch == None:
                fname = "Epoch_best"
            else:
                fname = f"Epoch_{self.opt.longVAE_load_epoch}"
            if self.opt.varying_length:
                if self.opt.eval_times != None:
                    fname += f"__eval_times_{self.opt.eval_times}"
                    self.opt.eval_times = self.opt.eval_times.split(",")
                
            self.opt.generated_dir = os.path.join('generated', 
                                                  self.opt.dataset_name,
                                                  f"{self.opt.run}--{self.opt.ID}",
                                                  fname)
            os.makedirs(self.opt.generated_dir, exist_ok=True)
            
        else:
            self.opt.splits = ['train', self.opt.eval_phase]
            signature = (str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-"))
            self.opt.ID = f"longVAE_training_{signature}"
        
        self.opt.savedir = os.path.join(self.opt.savedir, 'longVAE', self.opt.dataset_name, self.opt.ID)
        
        if not self.opt.no_train:
            os.makedirs(self.opt.savedir)
            # Save the training options for documentation
            save_options(self.opt)
        return self.opt   
        
        