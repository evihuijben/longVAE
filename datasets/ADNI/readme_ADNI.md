# Downloading and preparation of ADNI data

1. Download ADNI data from https://adni.loni.usc.edu/. The ID numbers that were
used for our work are provided in the file 'ADNI\_dataset\_details.csv'.

2. Apply preprocessing steps as described in the supplementary material of our work, this includes the following:
  * Applying N4 bias field correction using [ANTs](https://stnava.github.io/ANTs)
  * Register images to MNI152 (1 mm istotropic resolution) standard space using [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT)
  * Apply brain extraction using [HD-BET](https://github.com/MIC-DKFZ/HD-BET)
  * Clip the 99.99th percentile of voxel intensities within the brain.
  * Normalize image intensities to \[0, 1\].
  * Select axial slice number 100
  * Crop the slices to a size of 182 and pad to 192.
  * Define the ages of the subject at the time of acquisition. The first age is taken from the [ADNI website](https://adni.loni.usc.edu/) and the subsequent ages are calculated as floating point number based on the difference in acquisition dates.

3. Save the data in the files 'train.pt', 'val.pt', and 'test.pt' under the dictionary key 'data'. These files should contain a list with all the subjects, saved in the format [N\_observations, image_size, image_size\]. The Calculated ages should be saved in these files under the key 'times' as tensor elements in a list.

