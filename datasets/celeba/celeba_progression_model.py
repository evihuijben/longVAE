import numpy as np
import os
import torch
import random
import PIL.Image as Image
import copy
import torchvision.transforms as transforms
import argparse
import scipy.ndimage

############ Define Parameters ############
class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # Changeable parameters
        self.parser.add_argument('--dataroot', type=str, default='.', help="Path to where the original data is saved (.pt file per phase separately)")
        self.parser.add_argument('--save_folder', type=str, default='processed', help="Path to where the processed data will be saved")
        self.parser.add_argument('--splits', type=str, default='train,val,test', help="Which phases need to be processed")
        self.parser.add_argument('--sample_n', type=str, default='-1,-1,-1', help="How many images are used per set defined by splits. Use -1 when including all images in a set")
        
        # Dataset specific parameters
        self.parser.add_argument('--n_steps', type=int, default=10, help='Number of steps for a longitudinal dataset with a fixed number of observations per subject')
        self.parser.add_argument('--isize', type=int, default=64, help='Size of the image')
        
        # Progression model parameters        
        self.parser.add_argument('--sample_alpha_max', type=str, default='1,2', help="'min,max' for uniformly sampling intensity transform parameter alpha_max")
        self.parser.add_argument('--sample_beta', type=str, default='1,1.5', help="'min,max' for uniformly sampling growth factor beta")
        self.parser.add_argument('--sample_gamma', type=str, default='-90,90', help="'min,max' degrees for uniformly sampling rotation paramter gamma")
        
        # Base parameter
        self.parser.add_argument('--seed_value', type=int, default=0, help="Seed value for deterministic behavior")
        
        
        
    def parse(self):
        self.opt = self.parser.parse_args()
        
        # initialize lists which where entered as strings
        self.opt.splits = self.opt.splits.split(',')
        self.opt.sample_n = [int(x) for x in self.opt.sample_n.split(',')]
        
        self.opt.sample_alpha_max = [float(x) for x in self.opt.sample_alpha_max.split(',')]
        self.opt.sample_beta = [float(x) for x in self.opt.sample_beta.split(',')]
        self.opt.sample_gamma = [float(x) for x in self.opt.sample_gamma.split(',')]
        
        
        return self.opt


############ Helper functions ############
def set_seed(seed_value, pytorch=True):
    """
    Set seed for deterministic behavoir
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    if pytorch:
        torch.manual_seed(seed_value)

def make_grayscale(im):
    """
    Convert RGB image (3 channels) to grayscale (1 channel)
    """    
    if isinstance(im, np.ndarray):
        im = torch.from_numpy(im)
    im = transforms.functional.rgb_to_grayscale(im)[0]
    return np.array(im)

def change_intensities(im, delta):
    im = (im - im.min())/(im.max() - im.min())
    im = np.power(im, 1+delta)
    return im

def grow(im, delta_grow, isize):
    """
    Grow image
    """
    im = Image.fromarray(np.array(im))
    new_size = int(np.round(isize + delta_grow))
    im = im.resize((new_size, new_size))    
    return np.array(im)
    
def rotate(im, delta_rotate):
    """
    Rotate image
    """
    im = scipy.ndimage.rotate(im, delta_rotate, reshape=False, mode='nearest' )
    return  im
    
def crop_noise_normalize(im, isize, mean=0, sigma=0.001):
    """
    Crop the image to the given isize, add gaussian noise and normalize the
    image between 0 and 1.
    """
    # crop image
    start_crop = (im.shape[0]-isize)//2
    im = im[start_crop : start_crop + isize,
            start_crop : start_crop + isize,]
    
    # add gaussian noise
    gauss = np.random.normal(mean, sigma, im.shape)
    im = im + gauss
    
    # normalize image
    im = (im - im.min())/(im.max() - im.min())
    return im


def process(opt):
    """
    Process the images by first sampling a certain number of subjects, then 
    creating 'n_steps' progression time steps by converting the image to 
    grayscale, applying a non-linear intensity transform, applying a linear 
    growth, applying a linear rotation, cropping the image, adding gaussian 
    noise, and normalizing the image ([0,1]).
    """
    for phase, sample_n in zip(opt.splits, opt.sample_n):
        print(f">>> Processing {phase} set ...")
        
        x = torch.load(os.path.join(opt.dataroot, f"{phase}.pt"),
                       map_location=torch.device('cpu'))
        n_subjects = x['data'].shape[0]
        
           
        all_subjects = list(range(n_subjects))
        if sample_n!=-1 and sample_n<=len(all_subjects):
            all_subjects = all_subjects[:sample_n]
    
        sampled_set= x['data'][all_subjects]
        del x
        
        parameters = []
        final_tensor = torch.zeros((len(all_subjects),
                                    opt.n_steps, 
                                    opt.isize, 
                                    opt.isize), 
                                   dtype=torch.float32)
        
        for ind, subject_i in enumerate(all_subjects):
            if ind%100==0:
                print(f"\t{phase}: {ind}/{len(all_subjects)}")
            
            # Sample alpha (intensity transform parameter)
            final_alpha = np.random.uniform(opt.sample_alpha_max[0],
                                            opt.sample_alpha_max[1])
            # Sample new image size, based on beta (growthfactor)
            increase_n_pixels = np.random.randint(
                int(np.round(opt.isize * (opt.sample_beta[0]-1))),
                int(np.round(opt.isize * (opt.sample_beta[1]-1))))
            final_beta =  1 + increase_n_pixels/opt.isize
            
            # Sample gamma (rotation))
            final_gamma = np.random.uniform(opt.sample_gamma[0], 
                                            opt.sample_gamma[1])
            
            # Keep parameters for final save
            parameters.append({'subject_i': subject_i,
                               'final_alpha': final_alpha,
                               'final_beta': final_beta,
                               'final_gamma': final_gamma})
            
            # Create n_steps for every subject
            img = sampled_set[ind]
            img = (img - img.min())/(img.max() - img.min())
            for t_i in range(1, opt.n_steps +1):
                step = t_i * 1/opt.n_steps
                new_im = copy.deepcopy(img)
                
                new_im = make_grayscale(new_im)
                new_im = change_intensities(new_im, final_alpha * step)
                new_im = grow(new_im, increase_n_pixels * step, opt.isize)
                new_im = rotate(new_im, final_gamma * step)
                new_im = crop_noise_normalize(new_im, opt.isize)
                final_tensor[ind, t_i-1] = torch.from_numpy(new_im).float()
                
        # Save torch tensor
        os.makedirs(opt.save_folder, exist_ok=True)
        torch.save({'data': final_tensor,
                    'parameters': parameters},
                    os.path.join(opt.save_folder, f"{phase}.pt"))
        print(f'\t{phase} phase done.')
        
    
if __name__=='__main__':
    opt = Options().parse()
    set_seed(opt.seed_value)
    process(opt)
    

            
