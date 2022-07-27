import torch
import numpy as np
import os

def set_seed(seed_value, pytorch=True):
    """
    Set seed for deterministic behavior

    Parameters
    ----------
    seed_value : int
        Seed value.
    pytorch : bool
        Whether the torch seed should also be set. The default is True.

    Returns
    -------
    None.
    """
    import random
    random.seed(seed_value)
    np.random.seed(seed_value)
    if pytorch:
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True

def save_options(opt, find_run=False):
    """
    Save the training parameters in opt.savedir folder    

    Parameters
    ----------
    opt : argparser
        Parameters defining this run.
    find_run : bool, optional
        If True, find the last run where the options should be saved. The 
        default is False.

    Returns
    -------
    None.

    """
    if find_run:
        run = sorted(os.listdir(opt.savedir))[-1]
        fname = os.path.join(opt.savedir, run, 'opt.txt')
    else:
        fname = os.path.join(opt.savedir, 'opt.txt')
    args = vars(opt)
    s = '------------ Options -------------\n'
    for k, v in sorted(args.items()):
        s = s + f"{k}: {v}\n"
    s = s +'-------------- End ----------------'
    
    outf = open(fname, 'w')
    outf.write(s)
    outf.close()
    