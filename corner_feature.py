from skimage import color, feature
import numpy as np


def corner_feature(dataset):

    for image in dataset:

        im = color.rgb2gray(image[0])

        corners_map = feature.corner_shi_tomasi(im)

        image[0] = np.asarray(corners_map).reshape(-1)

    return dataset
