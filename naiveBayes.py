import numpy as np
from matplotlib import pyplot as plt
import os
import math

# This function takes in a directory and text labels file
# and reads data from both the directory and the .txt file
def load_data(dir_path, labels_text_file):
    # Length of the directory.
    imageCount = len(os.listdir(dir_path+'/'))
    # Read image as array and store the image size.
    imageSize = len(plt.imread(dir_path+'/1.jpg').flatten())
    # Initialize an array to store the input images.
    xN = np.zeros([imageCount, imageSize], dtype=float)
    # Initialize an array to store the labels.
    label = np.zeros([imageCount], dtype=int)

    # loop over the directory, divide each image by 255.0
    for i in range(1, imageCount+1):
        currentImage = plt.imread(dir_path + '/' + str(i) + '.jpg').flatten()
        xN[i-1] = currentImage/255.0

    # Read data from the labels text file.
    f=open(labels_text_file, "r")
    lines=f.readlines()
    count=0
    for line in lines:
        label[count]=int(line)
        count += 1
    # Find the variety of unique values.
    label_size=len(np.unique(label))
    # return the input image array, the label array, and the image size
    return xN, label, imageSize

# Load data from Train folder and Training Labels file.
train_data, train_label, image_size = load_data('Train',"Training Labels.txt")

# Load data from Test folder and Test Labels file.
test_data, test_labels, _ = load_data('Test',"Test Labels.txt")

# Determine the number of classes from the training label array.
classesNumbers = len(np.unique(train_label))
# Initialize an array for the each class mean.
classMean = np.zeros([classesNumbers, image_size], dtype=float)
# Initialize an array for each class variance
classVariance = np.zeros([classesNumbers, image_size], dtype=float)
# Initialize the empty confusion matrix
confusionMatrix = np.zeros([classesNumbers, classesNumbers],dtype=int)

# initialize and array for the probability that
# each input image belongs to a certain class
classProbability = np.zeros([classesNumbers, image_size], dtype=float)
splittedArray = np.split(train_data, classesNumbers)

# iterate over the classes and compute the mean and variance for each class
for i in range(classesNumbers):
    classMean[i] = np.mean(splittedArray[i], axis=0)
    classVariance[i] = ((np.std(splittedArray[i], axis=0)) ** 2)

# saturate values less than 0.01 to avoid NaN and INF values
classVariance[classVariance<0.01]=0.01

# compute the gaussian probability and find the maximum probability
# then increment the corresponding class in the confusion matrix.
for c in range(len(test_data)):
    classProbability = (1/np.sqrt(2 * math.pi * classVariance )) * np.exp(-((test_data[c] - classMean ) ** 2)/(2 * classVariance))
    res = np.argmax(np.prod(classProbability, axis=1))
    confusionMatrix[test_labels[c]][res] += 1
print(confusionMatrix)
print(confusionMatrix.shape)