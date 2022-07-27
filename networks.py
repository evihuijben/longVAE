import torch
import torch.nn as nn
import torch.nn.functional as F

from pythae.models.nn.base_architectures import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput

## CelebA encoder
class Encoder_VAE_CELEBA(BaseEncoder):
    """
    A Convolutional encoder neural net suited for CELEBA-64 (both grayschale and
    RGB) Variational Autoencoder-based models.
    """
    def __init__(self, args):
        BaseEncoder.__init__(self)
        self.input_dim = (args.n_channels, args.isize, args.isize)
        self.latent_dim = args.latent_dim
        self.n_channels = args.n_channels
        
        
        
        n_down = 4
        k, s, p = 4, 2, 1 # kernel, stride, padding
        in_channels = self.n_channels
        out_channels = 128
        new_size = args.isize
        layers = nn.ModuleList()
        for i in range(n_down):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, k, s, padding=p),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()))
            in_channels = out_channels
            out_channels *= 2
            new_size = int(((new_size + 2*p -k)/s) + 1)

        self.layers = layers
        self.depth = len(layers)
        self.embedding = nn.Linear(in_channels * new_size * new_size, args.latent_dim)
        self.log_var = nn.Linear(in_channels * new_size * new_size, args.latent_dim)
    def forward(self, x: torch.Tensor, output_layer_levels=None):
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input image.
        output_layer_levels : [int], optional
            The levels of the layers where the outputs are extracted. If None, 
            the last layer's output is returned.. The default is None.

        Returns
        -------
        output : An instance of ModelOutput
            An instance of ModelOutput containing the embeddings of the input data 
            under the key `embedding` and the **log** of the diagonal coefficient of the covariance 
            matrices under the key `log_covariance`. Optional: The outputs of the layers specified 
            in `output_layer_levels` arguments are available under the keys `embedding_layer_i` 
            where i is the layer's level.
        """
        output = ModelOutput()
        max_depth = self.depth
        if output_layer_levels is not None:
            assert all(self.depth >= levels > 0 or levels==-1 for levels in output_layer_levels), (
                f'Cannot output layer deeper than depth ({self.depth}). '\
                f'Got ({output_layer_levels}).'
                )
            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)
        out = x
        for i in range(max_depth):
            out = self.layers[i](out)
            if output_layer_levels is not None:
                if i+1 in output_layer_levels:
                    output[f'embedding_layer_{i+1}'] = out
        
            if i+1 == self.depth:
                output['embedding'] = self.embedding(out.reshape(x.shape[0], -1))
                output['log_covariance'] = self.log_var(out.reshape(x.shape[0], -1))
        return output
    
## Celeba Decoder
class Decoder_AE_CELEBA(BaseDecoder):
    """
    A Convolutional decoder neural net suited for CELEBA-64 (both grayschale and
    RGB) Autoencoder-based models.
    """
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)
        self.input_dim = (args.n_channels, args.isize, args.isize)
        self.latent_dim = args.latent_dim
        self.n_channels = args.n_channels
        
        layers = nn.ModuleList()
                
        layers.append(
            nn.Sequential(
                nn.Linear(args.latent_dim, 1024 * 8 * 8)
            )
        )
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, 5, 2, padding=2),
                nn.BatchNorm2d(512),
                nn.ReLU(),
            )
        )
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 5, 2, padding=1, output_padding=0),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
        )
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 5, 2, padding=2, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        )
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, self.n_channels, 5, 1, padding=1),
                nn.Sigmoid(),
            )
        )
        self.layers = layers
        self.depth = len(layers)
    def forward(self, z: torch.Tensor, output_layer_levels=None):
        """
        Forward pass

        Parameters
        ----------
        z : torch.Tensor
            DESCRIPTION.
        output_layer_levels : [int], optional
            The levels of the layers where the outputs are extracted. If None,
            the last layer's output is returned. The default is None.

        Returns
        -------
        output : An instance of ModelOutput
            An instance of ModelOutput containing the reconstruction of the 
            latent code under the key `reconstruction`. Optional: The outputs 
            of the layers specified in `output_layer_levels` arguments are 
            available under the keys 'reconstruction_layer_i' where i is the 
            layer's level.

        """
        output = ModelOutput()
        max_depth = self.depth
        if output_layer_levels is not None:
            assert all(self.depth >= levels > 0 or levels==-1 for levels in output_layer_levels), (
                f'Cannot output layer deeper than depth ({self.depth}). '\
                f'Got ({output_layer_levels}).'
                )
            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)
        out = z
        for i in range(max_depth):
            out = self.layers[i](out)
            if i == 0:
                out = out.reshape(z.shape[0], 1024, 8, 8)
            if output_layer_levels is not None:
                if i+1 in output_layer_levels:
                    output[f'reconstruction_layer_{i+1}'] = out
            if i+1 == self.depth:
                output['reconstruction'] = out
        return output

## ADNI Encoder
class Encoder_VAE_192(BaseEncoder):
    """
    A Convolutional encoder neural net suited for a Variational Autoencoder for
    one-channel 2D imaging data smaller than 1x192x192.
    """
    def __init__(self, args):
        BaseEncoder.__init__(self)
        # Define the amount of padding if the image size is smaller than 192
        desired_size = 192
        if args.isize < desired_size:
            pad1 = (desired_size - args.isize)//2
            pad2 = desired_size - args.isize - pad1
            self.padding = (pad1, pad2, pad1, pad2,)
        else:
            self.padding = None
        
        
        n_down = 5
        k, s, p = 4, 2, 1  # kernel, stride, padding
    
        in_channels = args.n_channels
        out_channels = 64     # starting number of encoder features
        layers = []
        new_size = desired_size
        for layer_i in range(n_down):
            layers.append(nn.Conv2d(in_channels, out_channels, k, s, p))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            
            in_channels = out_channels
            out_channels *= 2
            new_size = int(((new_size + 2*p -k)/s) + 1)
            
        self.layers = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_channels*new_size*new_size, args.latent_dim)
        self.fc_log_var = nn.Linear(in_channels*new_size*new_size, args.latent_dim)
    
    def forward(self, x):
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input image.

        Returns
        -------
        output : An instance of ModelOutput
            An instance of ModelOutput containing the embeddings of the input 
            data under the key `embedding` and the **log** of the diagonal 
            coefficient of the covariance matrices under the key `log_covariance`. 
            where i is the layer's level.

        """
        if self.padding:
            x = F.pad(x, self.padding)
            
        output = ModelOutput()
        out = self.layers(x)
        out = out.reshape(x.shape[0], -1)
        output['embedding'] = self.fc_mu(out)
        output['log_covariance'] = self.fc_log_var(out)
        return output
 
    
## ADNI Decoder
class Decoder_AE_192(BaseDecoder):
    """
    A Convolutional decoder neural net suited for a Autoencoder for one-channel
    2D imaging data smaller than 1x192x192.
    """
    def __init__(self, args):
        BaseDecoder.__init__(self)
        
        desired_size = 192
        if args.isize < desired_size:
            remove1 = (desired_size - args.isize)//2
            remove2 = desired_size - args.isize - remove1
            self.removing = (remove1, remove2, )
        else:
            self.removing = None
        
        
        
        n_down = 5
        k, s, p = 4, 2, 1  # kernel, stride, padding
    
        ngf = 64     # starting number of encoder features
        
        
        new_size = desired_size
        for layer_i in range(n_down):
            if layer_i > 0 :
                ngf *= 2
            new_size = int(((new_size + 2*p -k)/s) + 1)
        self.ngf = ngf
        self.final_size = new_size
        
        nfcf = ngf * self.final_size * self.final_size
        self.fc = nn.Sequential(
            nn.Linear(args.latent_dim, nfcf)
        )
        
        layers = []
        for layer_i in range(n_down-1):
            layers.append(nn.ConvTranspose2d(ngf, ngf//2, k, s, p))
            layers.append(nn.BatchNorm2d( ngf//2))
            layers.append(nn.ReLU())
            ngf = ngf//2
            
        layers.append(nn.ConvTranspose2d(ngf, args.n_channels, k, s, p))
        layers.append(nn.Sigmoid())
                
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, z):
        """
        Forward pass

        Parameters
        ----------
        z : torch.Tensor
            DESCRIPTION.

        Returns
        -------
        output : An instance of ModelOutput
            An instance of ModelOutput containing the reconstruction of the 
            latent code under the key `reconstruction`. 
        """
        output = ModelOutput()
        out = self.fc(z)
        out = out.reshape(z.shape[0], self.ngf, self.final_size, self.final_size)
        out = self.layers(out)
        if self.removing:
            out = out[:, :, self.removing[0]:-self.removing[1],
                      self.removing[0]: -self.removing[1]]
        output['reconstruction'] = out
        
        return output
