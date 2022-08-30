import os
import torch
import math
from torch.utils.data import DataLoader

class EmbeddingDataset(torch.utils.data.Dataset):
    # Dataset class for feature processing
    def __init__(self, embeddings, varying_length=False, **kwargs):
        self.data = embeddings
        
        self.varying_length = varying_length
        if self.varying_length:
            # Define observation times for sequences with varying length
            self.times = kwargs.pop('times')
        else:
            # Define mask if data is artificially removed from sequences
            # with a fixed number of observations
            self.mask = kwargs.pop('mask', None)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        x = self.data[index]
        sample = {'data': x}
        
        if self.varying_length:
            sample['times'] = self.times[index].squeeze(0).to(x.device)
        else:
            if self.mask == None:
                # When no mask is provided, all observations are considered
                mask = torch.ones((x.shape[0])).type(torch.float)
            if self.mask is not None:
                # Add a 1 for indiciating the mask at t0
                mask = torch.cat((torch.tensor([1.]).to(self.mask.device),
                                  self.mask[index]))
            sample['mask'] = mask.to(x.device)
        return sample

def normalize_times_for_varying_length(times):
    """
    Normalize observation times between 0 and 1 for the minimum and maximum
    observation time of the entire population.

    Parameters
    ----------
    times : [torch tensor]
        List of all the observation times per subject.

    Returns
    -------
    times : [torch tensor]
        List of all the normalized observation times per subject..

    """
    # Find minimum and maximum observation time
    max_time = max([torch.max(t) for t in times])
    min_time = min([torch.min(t) for t in times])
    # normalize times
    for i in range(len(times)):
        times[i] = (times[i] - min_time) / (max_time - min_time)
    return times
    
def extract_features(opt, vae_model, data):
    """
    Extract features by inputting the image data into the encoder of the VAE

    Parameters
    ----------
    opt : argparser
        Parameters defining this run.
    vae_model : torch neural network module
        VAE model defined by pythae.
    data : torch tensor
        Image data.

    Returns
    -------
    all_embeds : torch tensor
        Features obtained from encoding image data.

    """
    # Encode the data into embeddings using the pretrained VAE
    vae_model.to(opt.device)
    vae_model.eval()
    with torch.no_grad():

        all_embeds = []
        if opt.varying_length:
            for batch in data:
                inp = batch.reshape(-1,
                                    opt.n_channels,
                                    opt.isize,
                                    opt.isize).to(opt.device)
                this_embed = vae_model.encoder(inp).embedding.detach()
                all_embeds.append(this_embed.reshape(-1, vae_model.latent_dim))
                
        else:
            data = data.reshape(-1, opt.n_channels, opt.isize, opt.isize)            
            if opt.batchsize_VAE_eval == None:
                this_batchsize = data.shape[0]
            else:
                this_batchsize = opt.batchsize_VAE_eval

            all_embeds = []
            for batch_i in range(int(math.ceil(data.shape[0]/this_batchsize))):
                start = batch_i*this_batchsize
                stop = min([(batch_i+1)*this_batchsize, data.shape[0]])
                this_embed = vae_model.encoder(data[start : stop].type(torch.float).to(opt.device))
                all_embeds.append(this_embed.embedding.detach().reshape(-1, opt.n_steps, vae_model.latent_dim))
            all_embeds = torch.cat(all_embeds, dim=0)
    return all_embeds   
    
def load_data_longVAE(opt, vae_model):
    """
    

    Parameters
    ----------
    opt : argparser
        Parameters defining this run.
    vae_model : torch neural network module
        VAE model defined by pythae.

    Returns
    -------
    embedding_loaders : dict
        A dictionary with dataloaders containing the embedded features for all
        phases in opt.splits.

    """
    shuffles = {'train': True, 
                'val': False, 
                'test': False}
        
    embedding_loaders = {}
    for phase in opt.splits:
        if opt.varying_length:
            loaded = torch.load(os.path.join(opt.dataroot, f'{phase}.pt'),
                                map_location=opt.device)
            datatensor = loaded['data']
            times = normalize_times_for_varying_length(loaded['times'])
        else:
            if opt.missing_data_prob>=0 and opt.missing_data_prob<1:
                 # load data containing masked observations that were
                 # artificially removed by sampling
                loaded = torch.load(os.path.join(opt.dataroot, f'{phase}_missing_{opt.missing_data_prob}.pt'),
                                    map_location=opt.device)
                mask = loaded['mask']            
            else:
                # Load full dataset
                loaded = torch.load(os.path.join(opt.dataroot, f'{phase}.pt'),
                                    map_location=opt.device)
                mask = None
            datatensor = loaded['data']
        
        # Encode data into embeddings and define the dataset for the embeddings
        embeddings = extract_features(opt, vae_model, datatensor)
        if opt.varying_length:
            dataset = EmbeddingDataset(embeddings, opt.varying_length, times=times)
            this_batchsize = 1
        else:
            dataset = EmbeddingDataset(embeddings, opt.varying_length, mask=mask)
            this_batchsize = opt.batchsize

        # Create dataloader for embeddings
        embedding_loaders[phase] = DataLoader(dataset,
                                              batch_size=this_batchsize, 
                                              shuffle=shuffles[phase])
    return embedding_loaders