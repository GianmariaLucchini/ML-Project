# ML-Project

The Repository includes .py files, .ipynb files and the dataset.

The project is develped in Jupyter



### main.py

Load images
Extract features 
Split data (70% Training, 30% Test)
100 iterations with different Training/Test for CrossValidation 
Classify testSet for each feature (using Linear and Gaussian kernel)



### corner_feature.py

Extract corner detection feature for each image through the shi_tomasi function



### edge_feature.py

Extract edge detection feature for each image through the edge_detector function where compute the magnitude of the gradient for the map



### hog_feature.py

Extract Histogram of Oriented Gradients feature for each image through the hog function from the skimage library



### my_SVM.py

Create 'my_SVM_train' class to train our data
Create 'my_SVM_pred' class to predict vector labels and the accuracy
Create 'my_Kernel' to define the two different kernel
