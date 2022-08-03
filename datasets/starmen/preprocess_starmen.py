import numpy as np
import os
import torch


def load_images(root):
    # Load all images into torch tensors
    all_ims = []
    for im in sorted(os.listdir(root)):
        all_ims.append(torch.tensor(np.load(os.path.join(root, im))))
    tensor = torch.cat(all_ims).reshape(-1, 10, 64, 64)
    
    # Create training, validation and test split
    all_sets = {'train': tensor[:700],
                'val': tensor[700:900],
                'test': tensor[900:]}
    return all_sets

def save_tensors(all_sets):
    # Save all sets
    for phase in all_sets.keys():
        torch.save({'data': all_sets[phase]}, f'{phase}.pt')

def save_masked_sets(sets):
    # provided mask details:
    probability =  0.5
    
    fname = f'masks_missing_{probability}.pt'
    masks = torch.load(fname)
    
    for phase in ['train', 'val', 'test']:
        data = sets[phase]
        mask = masks[phase]
        
        expanded_mask = torch.cat(
            [torch.ones((mask.shape[0], 1, ),
                        dtype=torch.bool),
             mask],
            dim=-1).unsqueeze(2).unsqueeze(3)
        expanded_mask = expanded_mask.repeat((1, 1, data.shape[2], data.shape[3]))
        
        torch.save({'data': (expanded_mask*data),
                    'mask': mask.type(torch.float32)},
                   f'{phase}_missing_{probability}.pt')

if __name__=='__main__':
    # Follow the steps in 'readme_starmen.txt' before executing this file.
    dataroot = 'output_random/images'    
    
    # Load images
    print('>> Preprocessing starmen data ...')
    sets = load_images(dataroot)
    
    # save full dataset as tensors
    save_tensors(sets)
    
    # save masked dataset for training with missing data
    save_masked_sets(sets)
    
    print('>> Done')
