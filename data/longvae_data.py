import os
import torch
from torch.utils.data import DataLoader

    
# feature processing
class Dataset(torch.utils.data.Dataset):
    def __init__(self, digits, varying_length=False, **kwargs):
        self.varying_length = varying_length
        if self.varying_length:
            self.times = kwargs.pop('times')
        else:
            self.mask = kwargs.pop('mask', None)
        self.data = digits

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        x = self.data[index]
        sample = {'data': x}
        if self.varying_length:
            sample['times'] = self.times[index].squeeze(0).to(x.device)
        else:
            if self.mask is not None:
                mask = torch.cat((torch.tensor([1.]).to(self.mask.device),
                                  self.mask[index]))
            else:
                mask = torch.ones((x.shape[0])).type(torch.float)
            sample['mask'] = mask.to(x.device)
        return sample


def normalize_times_for_varying_length(times):
    # Times should be normalized between 0 and 1 for the minimum and maximum
    # observation time of the entire population.
    max_time = times[0][0]
    min_time = times[0][0]
    for i in range(len(times)):
        if torch.max(times[i]) > max_time:
            max_time = torch.max(times[i])
        if torch.min(times[i]) < min_time:
            min_time = torch.min(times[i])
    for i in range(len(times)):
        times[i] = (times[i] - min_time) / (max_time - min_time)
    return times

def extract_features(opt, vae_model, data):
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
            if opt.batchsize_VAE_eval == None:
                this_batchsize = data.shape[0]
            else:
                this_batchsize = opt.batchsize_VAE_eval
            
            data = data.reshape(-1, opt.n_channels, opt.isize, opt.isize)
            this_loader = DataLoader(data.type(torch.float).to(opt.device), 
                                 batch_size=this_batchsize, 
                                 shuffle=False)
            for batch in this_loader:
                this_embed = vae_model.encoder(batch).embedding.detach()
                all_embeds.append(this_embed.reshape(-1, 
                                                     opt.n_steps, 
                                                     vae_model.latent_dim))
                
            all_embeds = torch.cat(all_embeds, dim=0)
    return all_embeds
    

def load_data_longVAE(opt, vae_model):
    shuffles = {'train': True, 
                'val': False, 
                'test': False}
    
    bs = {'train': opt.batchsize_train, 
          'val': opt.batchsize_eval,
          'test': opt.batchsize_eval}

        
    embedding_loaders = {}
    for phase in opt.splits:
        if opt.varying_length:
            loaded = torch.load(os.path.join(opt.dataroot, f'{phase}.pt'))
            datatensor = loaded['data']
            times = normalize_times_for_varying_length(loaded['times'])
        else:
            if phase == 'train' and opt.missing_data_prob>=0 and opt.missing_data_prob<1:
                loaded = torch.load(os.path.join(opt.dataroot, f'{phase}_missing_{opt.missing_data_prob}.pt'))
                mask = loaded['mask']            
            else:
                loaded = torch.load(os.path.join(opt.dataroot, f'{phase}.pt'))
                mask = None
            datatensor = loaded['data']
        
        # Encode data into embeddings and define the dataset for the embeddings
        embeddings = extract_features(opt, vae_model, datatensor)
        if opt.varying_length:
            dataset = Dataset(embeddings, opt.varying_length, times=times)
            this_batchsize=1
        else:
            dataset = Dataset(embeddings, opt.varying_length, mask=mask)
            if  bs[phase] == None:
                this_batchsize = embeddings.shape[0]
            else:
                this_batchsize =  bs[phase]
                
        # Create dataloader for embeddings
        embedding_loaders[phase] = DataLoader(dataset,
                                              batch_size=this_batchsize, 
                                              shuffle=shuffles[phase])
        
    return embedding_loaders


    
