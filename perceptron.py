import numpy as np
from matplotlib import pyplot as plt
import os

# Function to intialize the initial weight vector
def initialize_weightVector(labels_size, image_size):
    weight = np.zeros([labels_size, image_size+1], dtype=float)
    for i in range(len(weight)):
        weight[i][0] = 1
    return weight 

# Function to read data from given directory and
# given text file path.
def load_data(dir_name, label_filename):
    #2400
    number_images = len(os.listdir(dir_name+'/'))
    #784
    image_size = len(plt.imread(dir_name+'/1.jpg').flatten())
    # Input images as numpy array
    xN = np.zeros([number_images, image_size+1], dtype=int)
    # Labels as numpy array
    labels = np.zeros([number_images], dtype=int)


    for i in range(1, number_images+1):
        current_image = np.append(plt.imread(dir_name+'/' + str(i)+'.jpg').flatten(), [1])
        xN[i - 1] = current_image
    
    # Read from labels file 
    file = open(label_filename, 'r')
    lines = file.readlines()
    i = 0
    for line in lines:
        labels[i] = int(line)
        i += 1
    label_size = len(np.unique(labels))
    return xN, labels, label_size, image_size

# Function to update the weight vector if the point is missclassified
def update_missclassified_points(point, weights, label, learning_rate):
    for i in range(len(weights)):
        p = np.dot(weights[i], np.transpose(point))
        if(i==label):
            if(p < 0):
                weights[i] += learning_rate * point * 1
        else:
            if(p >= 0):
                weights[i] += learning_rate * point * -1
    return weights

# Testing for Training images and Training's labels
train_data, train_labels, label_size, image_size = load_data('Train', 'Training Labels.txt')

# Testing for Test images and Test's labels
test_data, test_labels, _, _ = load_data('Test', 'Test Labels.txt')

# try for each learning rate and build confusion matrix
for learn_rate in range(0, 10):
    # initialize the weight vector
    weights = initialize_weightVector(label_size, image_size)
    # intialize the confusion matrix
    confusion = np.zeros([len(np.unique(train_labels)), len(np.unique(train_labels))], dtype=int)

    # For classifier loops over images 500 iterations
    for i in range(500):
        for img in range(len(train_data)):
            # update the weight vector
            weights = update_missclassified_points(train_data[img], weights, train_labels[img], 10 ** (-(learn_rate)))
    

    for j in range(test_labels.size):
        res = np.argmax(np.dot(weights, test_data[j]))
        confusion[test_labels[j]][res]+=1
        # Print the confusion Matrix for all learning rates.
        print('Learning rate ' + str(learn_rate) + '\n', confusion)
        # Save the Confusion matrix as an image
        # For each Learning rate
    plt.imsave('Confusion-'+ ' '+ str(learn_rate)+ '.jpg', confusion)
    print("Confusion Matrix-" + str(learn_rate)+ ' is saved')