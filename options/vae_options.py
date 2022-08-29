from .base_options import BaseOptions

import os
class VAEOptions(BaseOptions):
    """
    This class includes the options for training the VAE (step 1). It also 
    includews shared options defined in BaseOptions.
    """
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--network_name', type=str, default='default', help="network structure: ['default', 'celeba', 'adni']")
        self.parser.add_argument('--latent_dim', type=int, default=8, help="Number of features in the latent space")

    def parse(self):
        """
        Parse arguments and initialize them

        Returns
        -------
        opt : argparser
            Parameters defining this run.

        """
        self.opt = self.parser.parse_args()
        self.initialize()
        
        self.opt.savedir = os.path.join(self.opt.savedir, 'VAE', self.opt.dataset_name)
        return self.opt