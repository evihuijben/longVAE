from options.vae_options import VAEOptions
from data.vae_data import load_dataset
from utils import set_seed, save_options

from pythae.models import BetaVAE, BetaVAEConfig
from pythae.trainers import BaseTrainerConfig            
from pythae.pipelines import TrainingPipeline

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
        train_config = BaseTrainerConfig(batch_size=opt.batchsize_train,
                                          num_epochs=opt.n_epochs,
                                          learning_rate=opt.lr,
                                          output_dir = self.opt.savedir)


        # Initialize neural networks
        if self.opt.network_name.lower() == 'default':
            self.vae_model = BetaVAE(model_config=vae_model_config)
        else:
            if self.opt.network_name.lower() == 'celeba':
                from networks import Encoder_VAE_CELEBA as Encoder
                from networks import Decoder_AE_CELEBA as Decoder
            elif self.opt.network_name.lower() == 'adni':
                from networks import Encoder_VAE_192 as Encoder
                from networks import Decoder_AE_192 as Decoder
            else:
                raise NotImplementedError(
                    f"Network {self.opt.network_name} is unknown,"\
                    "choose from ['default', 'celeba', 'adni']")
            self.vae_model = BetaVAE(model_config=vae_model_config,
                                     encoder=Encoder(self.opt),
                                     decoder=Decoder(self.opt))
        
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
    for phase in dataset.keys():
        if dataset[phase].datatensor != None:
            dataset[phase] = dataset[phase].datatensor
            
    # Initialize model and start training
    my_model = Model(opt)
    my_model.training_pipeline(train_data=dataset['train'],
                                eval_data=dataset[opt.eval_phase])
    
    # Save the training options for documentation
    save_options(opt, find_run=True)


        

            
            
            
    