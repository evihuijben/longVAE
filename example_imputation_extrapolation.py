from options.longvae_options import LongVAEOptions
from data.longvae_data import load_data_longVAE
from model import LongVAE
from utils import set_seed
import os

from pythae.models import BetaVAE
import torch


def synthetize_images(opt, vae_model, longvae_model, embedding_loaders):
    vae_model = vae_model.to(opt.device)
    longvae_model = longvae_model.to(opt.device)
    
    vae_model.eval()
    longvae_model.eval()
    with torch.no_grad():
        for phase in opt.splits:
            all_ims = []
            for batch_i, batch in enumerate(embedding_loaders[opt.eval_phase]):                
                if opt.varying_length:
                    if opt.eval_times == None:
                        eval_times = batch['times']
                    else:
                        eval_times = torch.FloatTensor([float(t) for t in opt.eval_times])
                        eval_times = eval_times.to(opt.device)
                    recon_traj = longvae_model.extrapolate_traj(batch['data'], 
                                                                visit_time=batch['times'],
                                                                eval_visit_time=eval_times)
                    recon_traj_dec = vae_model.decoder(recon_traj).reconstruction.reshape(-1, opt.isize, opt.isize)
                else:
                    recon_traj = longvae_model.extrapolate_traj(batch['data'], mask=batch['mask'])
                    recon_traj_dec = vae_model.decoder(recon_traj).reconstruction.reshape(-1, opt.n_steps, opt.isize, opt.isize)
                
                all_ims.append(recon_traj_dec.detach())
                
            if opt.varying_length:
                save_dict = {'syn_data': all_ims,
                             'eval_times': eval_times}
            else:
                save_dict = {'syn_data': torch.cat(all_ims, dim=0)}
            
            
            torch.save(save_dict, os.path.join(opt.generated_dir, 
                                               'Synthetic_images.pt'))
            
            print(f'Synthetic images saved in the folder {opt.generated_dir}')


if __name__=='__main__':
    # Load training options
    opt = LongVAEOptions().parse()

    # Set seed for deterministic behavior
    set_seed(opt.seed_value)
    
    # load pretrained VAE model
    vae_model = BetaVAE.load_from_folder(opt.trained_VAE_path).to(opt.device)
    vae_model = vae_model.eval()
    
    # Load data and encode into embedinngs
    embedding_loaders = load_data_longVAE(opt, vae_model)    
    
    # Initialize and train longVAE model
    longvae_model = LongVAE(opt)
    if opt.longVAE_load_epoch == None:
        fname = 'final_model.pt'
    else:
        fname = f'final_model_{opt.longVAE_load_epoch}.pt'
    trained_longVAE_path = os.path.join(opt.savedir, fname)
    longvae_model.load_state_dict(torch.load(trained_longVAE_path)['model_state_dict'])
    
    synthetize_images(opt, vae_model, longvae_model, embedding_loaders)
    
    