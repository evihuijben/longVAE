from options.vae_options import VAEOptions
from data.vae_data import load_dataset
from utils import set_seed, save_options

from pythae.models import BetaVAE, BetaVAEConfig
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig 

## MODEL
class Model():
    def __init__(self, opt):
        self.opt = opt

        # Define VAE configuration
        vae_model_config = BetaVAEConfig(input_dim= (opt.n_channels, 
                                               opt.isize, 
                                               opt.isize),
                                  latent_dim = opt.latent_dim,
                                  beta=opt.beta)
        
        # Define training configuration
        train_config = BaseTrainerConfig(batch_size=opt.batchsize,
                                          num_epochs=opt.n_epochs,
                                          learning_rate=opt.lr,
                                          output_dir = self.opt.savedir)


        # Initialize neural networks
        if self.opt.network_name.lower() == 'default':
            self.vae_model = BetaVAE(model_config=vae_model_config)
        else:
            if self.opt.network_name.lower() == 'celeba':
                from networks import Encoder_VAE_CELEBA
                from networks import Decoder_AE_CELEBA
                decoder = Decoder_AE_CELEBA(self.opt)
                encoder = Encoder_VAE_CELEBA(self.opt)
            elif self.opt.network_name.lower() == 'adni':
                from networks import Encoder_VAE_192
                from networks import Decoder_AE_192
                encoder = Encoder_VAE_192(self.opt)
                decoder = Decoder_AE_192(self.opt)
            else:
                raise NotImplementedError(
                    f"Network {self.opt.network_name} is unknown,"\
                    "choose from ['default', 'celeba', 'adni']")
            self.vae_model = BetaVAE(model_config=vae_model_config,
                                     encoder=encoder,
                                     decoder=decoder)
        
        # Define training pipeline
        self.training_pipeline = TrainingPipeline(model=self.vae_model,
                                                  training_config=train_config)


if __name__=='__main__':
    # Load training options
    opt = VAEOptions().parse()

    # Set seed for deterministic behavior
    set_seed(opt.seed_value)
    
    # load data
    dataset = load_dataset(opt)

    # Initialize model and start training
    my_model = Model(opt)
    my_model.training_pipeline(train_data=dataset['train'],
                                eval_data=dataset[opt.eval_phase])
    
    # Save the training options for documentation
    save_options(opt, find_run=True)