# Downloading and processing CelebA data

1. The dataset used for this paper was the cropped and aligned ($64\times64$) downloaded in 2021 from the [original website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)


2. Save the RGB images in a 'train.pt', 'val.pt', and 'test.pt' file as a torch tensor under the key 'data' in the format of \[N\_subjects, 3, 64, 64\].

3. To create the longitudinal data as described in the paper, run:
```
python datasets/celeba/celeba_progression_model.py
```

