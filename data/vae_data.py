import os
import torch
        
def load_dataset(opt):
    """
    Define datasets for all phases defined by opt.splits

    Parameters
    ----------
    opt : argparser
        Parameters defining this run.

    Returns
    -------
    dataset : dict
        Dataset objects for every phase defined by opt.splits.

    """
    dataset = {}
    for phase in opt.splits:
        if opt.varying_length:
            # Load datatensor list containing elements of shape [seq_length, isize, isize]
            datatensor = torch.load(os.path.join(opt.dataroot, f'{phase}.pt'),
                                    map_location=opt.device)['data'] 
            # Concatenate all images since longitudinal info is not needed for the VAE
            datatensor = torch.cat(datatensor, dim = 0)
        else:
            if opt.missing_data_prob>=0 and opt.missing_data_prob<1:
                fname = os.path.join(opt.dataroot, f'{phase}_missing_{opt.missing_data_prob}.pt')
            else:
                fname = os.path.join(opt.dataroot, f'{phase}.pt')
            assert os.path.isfile(fname), "Dataset file '{fname}' does not exist. Please follow the described preprocessing steps."

            # Load data and mask
            loaded = torch.load(fname, map_location=opt.device)
            datatensor = loaded['data']
        
            if opt.missing_data_prob>=0 and opt.missing_data_prob<1:
                mask = loaded['mask']
                
                # Remove masked image from the training data
                not_missing = []
                for i in range(datatensor.shape[0]):
                    for j in range(datatensor.shape[1]):
                        if torch.sum(datatensor[i][j]).item() > 0:
                            not_missing.append(datatensor[i][j].reshape(opt.n_channels,
                                                                        opt.isize*opt.isize))
                datatensor = torch.cat(not_missing, dim=0)

            # Reshape data to the desired size
            datatensor = datatensor.reshape(-1, opt.n_channels, opt.isize, opt.isize).type(torch.float32).to(opt.device)


        dataset[phase] = datatensor
        print(f"{phase}: total number of images: {dataset[phase].shape[0]}")
    print('>> Data loaded')
    return dataset