import numpy as np
import os
import torch


if __name__=='__main__':
    # Follow the steps in 'readme_starmen.txt' before executing this file
    
    # Load images
    PATH = 'output_random/images'
    imgs = sorted(os.listdir(PATH))
    all_ims = []
    for im in sorted(os.listdir(PATH)):
        all_ims.append(torch.tensor(np.load(os.path.join(PATH, im))))
    tensor = torch.cat(all_ims).reshape(-1, 10, 64, 64)
    
    # Save training, validation and test set
    torch.save({'data': tensor[:700]}, 'train.pt')
    torch.save({'data': tensor[700:900]}, 'val.pt')
    torch.save({'data': tensor[900:]}, 'test.pt')
    
