from options.longvae_options import LongVAEOptions
from data.longvae_data import load_data_longVAE
from models import LongVAE
from utils import set_seed


from pythae.models import BetaVAE
import torch
import torch.optim as optim


def train(model, opt, dataloaders):
    best_loss, best_model = 1e10, None
    
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    # define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, factor=0.5, patience=200, verbose=True)
    
    model.to(opt.device)
    for epoch in range(opt.n_epochs):
        train_loss = train_step(model, opt, optimizer, dataloaders['train'], epoch)    
        eval_loss = eval_step(model, opt, dataloaders[opt.eval_phase], epoch)
        scheduler.step(eval_loss)
        
        # define best model on validation set
        if eval_loss < best_loss:
            best_model, best_loss = model, eval_loss
            print(' >> Best val epoch:', epoch)
            save_weights(model, opt, epoch, best=True)                            
            
        # output training information and save intermediate model weights
        # if epoch % 20 == 0:
        if epoch % 1 == 0:
            print(f'Epoch {epoch}: Train loss: {train_loss}\t Eval loss: {eval_loss}')
        if epoch % opt.save_freq == 0:
            save_weights(model, opt, epoch, best=False)
    return best_model


def train_step(model, opt, optimizer, train_loader, epoch):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(train_loader):
        if opt.varying_length:
            # Include time stampes for data with varying sequence lengths.
            output = model(batch['data'], epoch=epoch, visit_time=batch['times'])
        else:
            # Include a mask for data with a fixed sequence length.
            # This masked is used for cases where observations are artificially
            # removed.
            output = model(batch['data'], epoch=epoch, mask=batch['mask'])
        loss = output.loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    return epoch_loss


def eval_step(model, opt, eval_loader, epoch):
    model.eval()
    epoch_loss = 0
    for i, batch in enumerate(eval_loader):
        if opt.varying_length:
            # Include time stampes for data with varying sequence lengths.
            output = model(batch['data'], epoch=epoch, visit_time=batch['times'])
        else:
            # Include a mask for data with a fixed sequence length.
            # This masked is used for cases where observations are artificially
            # removed.
            output = model(batch['data'], epoch=epoch, mask=batch['mask'])
        epoch_loss += output.loss.item()
    epoch_loss /= len(eval_loader)
    return epoch_loss


def save_weights(model, opt, epoch, best=True):
    if best == True:
        fname = f"{opt.weights_dir}.pt"
    else:
        fname = f"{opt.weights_dir}_{epoch}.pt"
        
    # hier, kan opt weg?
    save_dict = {'model_state_dict': model.state_dict(),
                'training_config': opt,
                'epoch': epoch}
    torch.save(save_dict, fname)


if __name__=='__main__':
    # Load training options
    opt = LongVAEOptions().parse()
    
    # Set seed for deterministic behavior
    set_seed(opt.seed_value)
    
    # load pretrained VAE model
    vae_model = BetaVAE.load_from_folder(opt.trained_VAE_path).cuda()
    vae_model = vae_model.eval()
    
    # Load data and encode into embedinngs
    embedding_loaders = load_data_longVAE(opt, vae_model)    
    
    # Initialize and train longVAE model
    model = LongVAE(opt)
    trained_model = train(model=model, 
                          opt = opt,
                          dataloaders=embedding_loaders)
    