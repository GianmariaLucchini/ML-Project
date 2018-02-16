# # CITYSCAPE VS. LANDSCAPE

import numpy as np
import os
import glob
import random
import load_images
import hog_feature
import edge_feature
import corner_feature
import my_SVM

def split_data(dataset):

    random.shuffle(dataset)
    datasetX = []
    datasetY = []

    for n in dataset:
        datasetX.append(n[0])
        datasetY.append([n[1]])

    # 70% Training, 30% Test

    datasetX_testSplit = int(.7 * len(datasetX))
    datasetX_training = datasetX[:datasetX_testSplit]
    datasetX_test = datasetX[datasetX_testSplit:]

    datasetY_testSplit = int(.7 * len(datasetY))
    datasetY_training = datasetY[:datasetY_testSplit]
    datasetY_test = datasetY[datasetY_testSplit:]

    return datasetX_training, datasetY_training, datasetX_test, datasetY_test


def my_classifier(kernel_type, dataset):

    average = 0

    # CrossValidation

    for k in range(1,100):

        trainingX, trainingY, testX, testY = split_data(dataset)

        if kernel_type == "gaussian":
            clf = my_SVM.my_SVM_train("gaussian", my_SVM.my_Kernel.gaussian(50))
        else:
            clf = my_SVM.my_SVM_train("linear", my_SVM.my_Kernel.linear())

        _clf = clf.train(np.asarray(trainingX), np.asarray(trainingY))
        average += _clf.score(_clf.predict(np.asarray(testX)), np.asarray(testY))

    return average/100


# Loading Images

dataset = load_images.load()

# Extracting Features

corner_dataset = corner_feature.corner_feature(dataset)
edge_dataset = edge_feature.edge_feature(load_images.load())
hog_dataset = hog_feature.hog_feature(load_images.load())




# Corner

# Linear Kernel -> Accuracy  64.26%
print(my_classifier("linear", corner_dataset))

# Gaussian Kernel -> Accuracy  52.41%
print(my_classifier("gaussian", corner_dataset))



# Edge

# Linear Kernel -> Accuracy  56.25%
print(my_classifier("linear", edge_dataset))

# Gaussian Kernel -> Accuracy  52.06%
print(my_classifier("gaussian", edge_dataset))



# HOG

# Linear Kernel -> Accuracy  80.19%
print(my_classifier("linear", hog_dataset))

# Gaussian Kernel -> Accuracy  53.33%
print(my_classifier("gaussian", hog_dataset))
