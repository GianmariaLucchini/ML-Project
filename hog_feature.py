from skimage.feature import hog
from skimage import color
import numpy as np


def hog_feature(dataset):

    for image in dataset:

        im = color.rgb2gray(image[0])

        aux, hog_image = hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True, block_norm='L2-Hys')

        image[0] = np.asarray(hog_image).reshape(-1)

    return dataset
