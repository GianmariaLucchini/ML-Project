from skimage import color, filters
import numpy as np


def edge_detector(img, sigma, threshold):
    # Convert to grayscale and convert the image to float
    _img = color.rgb2gray(img)

    # Apply Gaussian filter
    img_smooth = filters.gaussian(_img, sigma)

    # Compute first derivatives with the following kernel
    k = [-0.5,0,0.5]

    # Compute first derivative along x
    Ix = np.copy(img_smooth)
    for i in range(img_smooth.shape[0]):
        Ix[i,:] = np.convolve(img_smooth[i,:], k, mode='same')

    # Compute first derivative along y
    Iy = np.copy(img_smooth)
    for j in range(img.shape[1]):
        Iy[:,j] = np.convolve(img_smooth[:,j], np.transpose(k), mode='same')

    # Compute the mangnitude of the gradient
    G = np.sqrt(np.square(Ix) + np.square(Iy))

    # Generate edge map
    edge = G > threshold

    return edge


# In[3]:


def edge_feature(dataset):

    for image in dataset:

        edge = edge_detector(image[0], 1, 0.035)

        image[0] = np.asarray(edge).reshape(-1)

    return dataset
