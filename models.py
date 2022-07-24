import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple, Any

class ModelOutput(OrderedDict):
    """Base ModelOutput class fixing the output type from the models. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class LongVAE(nn.Module):
    def __init__(self,  opt):
        nn.Module.__init__(self)
        
        self.varying_length = opt.varying_length
        
        if not self.varying_length:
            self.n_steps = opt.n_steps
            input_size_RNN = opt.latent_dim
            input_size = opt.latent_dim
        else:
            input_size_RNN = opt.latent_dim + 1
            input_size = opt.latent_dim
        
        self.beta = opt.beta
        self.warmup_epoch = 0
  
        self.log_var_eta_prior = torch.log(torch.tensor([float(opt.LongVAE_priors)], requires_grad=False)).to(opt.device)
        self.log_var_tau_prior = torch.log(torch.tensor([float(opt.LongVAE_priors)], requires_grad=False)).to(opt.device)
        
        self.rnn = nn.RNN(input_size=input_size_RNN,
                          hidden_size=opt.hidden, 
                          num_layers=3)
        
        self.fc_lbd = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128), 
            nn.ReLU()
        )
        self.fc = nn.Linear(opt.hidden, 128)

        self.mu_lbd = nn.Linear(128, opt.latent_dim-1)
        self.log_var_lbd = nn.Linear(128, opt.latent_dim-1)
        self.mu_eta = nn.Linear(128, 1)
        self.log_var_eta = nn.Linear(128, 1)
        self.mu_tau = nn.Linear(128, 1)
        self.log_var_tau = nn.Linear(128, 1)

        self.decoder = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_size)
        )

    def step(self, x, visit_time=None, eval_visit_time=None):
        batch_size = x.shape[0]
        if self.varying_length:
            rnn_input = torch.hstack((x.squeeze(), 
                                      visit_time.reshape(x.shape[-2], 1).type(x.dtype))
                                     ).unsqueeze(0)
            
            
        else:
            visit_time = torch.linspace(0, 1, self.n_steps).to(x.device)
            eval_visit_time = visit_time
            rnn_input = x
            
        out, hidden = self.rnn(rnn_input)   
        fc_out = self.fc(out[:, -1, :])
        fc_out_lbd = self.fc_lbd(x[:, 0, :])
                
        mu_lbd = self.mu_lbd(fc_out_lbd)
        log_var_lbd = self.log_var_lbd(fc_out_lbd)
        std_lbd = torch.exp(0.5 * log_var_lbd)
        lbd = self._sample_gauss(mu_lbd, std_lbd)
        
        mu_eta = self.mu_eta(fc_out)
        log_var_eta = self.log_var_eta(fc_out)
        std_eta = torch.exp(0.5 * log_var_eta)
        eta = self._sample_gauss(mu_eta, std_eta)
        
        mu_tau = self.mu_tau(fc_out)
        log_var_tau = self.log_var_tau(fc_out)
        std_tau = torch.exp(0.5 * log_var_tau)
        tau = self._sample_gauss(mu_tau, std_tau)
                

        l = eta.exp()* (eval_visit_time - tau) # [B, T]
        
        l_t = torch.cat((l.unsqueeze(1), 
                         lbd.unsqueeze(-1).expand(-1, -1, len(eval_visit_time))
                         ), dim=1) # [B, D, T]
        
        l_t_reshaped = l_t.movedim(2, 1).reshape(batch_size*len(eval_visit_time), -1) # [BxT, D]
        
        recon_x = self.decoder(l_t_reshaped)
        
        features = {'lbd': {'mu': mu_lbd, 'logvar': log_var_lbd, 'sampled': lbd},
                    'tau': {'mu': mu_tau, 'logvar': log_var_tau, 'sampled': tau},
                    'eta': {'mu': mu_eta, 'logvar': log_var_eta, 'sampled': eta}
                    }
        return recon_x, features

    def forward(self, x, **kwargs):
        epoch = kwargs.pop('epoch', self.warmup_epoch)
        
        output = ModelOutput()
        if self.varying_length:
            visit_time = kwargs.pop('visit_time').squeeze().type(x.dtype)
            recon_x, features = self.step(x, visit_time, visit_time)
        else:
            recon_x, features = self.step(x)
        mask = kwargs.pop('mask', torch.tensor([1]).to(x.device))
        loss = self.loss_function(recon_x, x, features, epoch, mask)

        output = ModelOutput(
            loss=loss,
            recon_x=recon_x,
            mu_lbd=features['lbd']['mu'],
            log_var_lbd=features['lbd']['logvar'],
            mu_eta=features['eta']['mu'],
            log_var_eta=features['eta']['logvar'],
            mu_tau=features['tau']['mu'],
            logvar_eta=features['tau']['logvar'],
        )

        return output

    def extrapolate_traj(self, x, **kwargs):
        """Expect x of shape [batch, time, dim]"""
        if self.varying_length:
            visit_time = kwargs.pop('visit_time')
            eval_visit_time = kwargs.pop('eval_visit_time')
            recon_x, _ = self.step(x, visit_time, eval_visit_time)
        else:
            recon_x, _ = self.step(x)
        return recon_x
    

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self, recon_x, x, features, epoch, mask):
        mu_lbd, log_var_lbd = features['lbd']['mu'], features['lbd']['logvar']
        mu_tau, log_var_tau = features['tau']['mu'], features['tau']['logvar']
        mu_eta, log_var_eta = features['eta']['mu'], features['eta']['logvar']

        if self.varying_length:
            recon_loss = F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                ).sum(dim=-1)
        else:
            mask = mask.unsqueeze(-1)                
            recon_loss = F.mse_loss(
                    (mask*recon_x.reshape(x.shape[0], x.shape[1], -1)).reshape(x.shape[0], -1),
                    (mask*x).reshape(x.shape[0], -1),
                    reduction="none",
                ).sum(dim=-1)
            

        KLD_lbd = -0.5 * torch.sum(1 + log_var_lbd - mu_lbd.pow(2) - log_var_lbd.exp(), dim=-1)
        KLD_eta = -0.5 * torch.sum(1 + log_var_eta - self.log_var_eta_prior - (mu_eta.pow(2) + log_var_eta.exp())/ self.log_var_eta_prior.exp() , dim=-1)
        KLD_tau = -0.5 * torch.sum(1 + log_var_tau - self.log_var_tau_prior - (mu_tau.pow(2) + log_var_tau.exp())/ self.log_var_tau_prior.exp(), dim=-1)


        beta = min(self.beta, self.beta * epoch / (self.warmup_epoch+1) )

        return (recon_loss + beta * (KLD_lbd + KLD_eta + KLD_tau)).mean(dim=0)