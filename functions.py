import numpy as np

'''
Source: https://github.com/smajida/mnist-fun
additional_functions.py
'''
def add_gaussian_noise(image, mean, stddev):
    n_imgs = image.shape[0]
    n_rows = image.shape[1]
    n_cols = image.shape[2]
    if stddev == 0:
        noise = np.zeros((n_imgs, n_rows, n_cols))
    else:
        noise = np.random.normal(mean, stddev/255.,
                                 (n_imgs, n_rows, n_cols))
    noisy_x = image + noise
    clipped_noisy_x = np.clip(noisy_x, 0., 1.)
    return clipped_noisy_x
