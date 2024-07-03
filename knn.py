"""

Author: Anjola Aina
Date: February 12th, 2024

Description: This file implements the K-nearest neighbour algorithm.

It uses the CSV dataset from: https://github.com/dylan-slack/TalkToModel

Information about Feature Scaling:

The standardization method was chosen as it provided the best accuracy for the KNN
classifier.

Max-min scaling, normalization, and robust scaling were implemented, and provided 0.50 accuracy for the classifier, as opposed to the 0.51 accuracy with standardization.
 
The following code below shows the means used to implement max-min scaling, normalization, and robust scaling:

Max-min (0.46 accuracy): 

x_train = ((x_train - x_train.min()) / (x_train.max() - x_train.min()))
x_test = ((x_test - x_train.min()) / (x_train.max() - x_train.min()))

Normalization (0.45 accuracy):

x_train = (x_train - x_train.mean() / x_train.std())
x_test = (x_test - x_train.mean() / x_train.std())

Robust Scalar (0.65 accuracy - BEST):

x_train_q1, x_train_q3 = np.percentile(x_train, [25, 75])
x_train_iqr = x_train_q3 - x_train_q1
x_train = (x_train - np.median(x_train) / x_train_iqr)
x_test = (x_test - np.median(x_train) / x_train_iqr)

Out of these three options, the robust scalar produced the best results with accuracy, and was used in this implementation of the K-nearest neighbour algorithm.
We can assume that this dataset had a lot of outliers which both normalization and max-min are affected by.

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load the df model 
df = pd.read_csv('diabetes.csv')

# extracting the x and y columns from the dataframe
y = df['y']
x = df.drop('y', axis='columns')

# creating a matrix x of input and vector y of expected outputs for the classifier
x = x.values
y = y.values

# dividing the dataset by allocating 20%% (test_size = 0.2) for testing, and 80% for training
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)

# applying feature standardization using robust scalar
x_train_q1, x_train_q3 = np.percentile(x_train, [25, 75])
x_train_iqr = x_train_q3 - x_train_q1
x_train = (x_train - np.median(x_train) / x_train_iqr)
x_test = (x_test - np.median(x_train) / x_train_iqr)

def eucilean_distance(a, b):
    """
    Implements the eucilean distance used to calcuate the distance between 
    two values.
    
    Params:
        - a - the first value
        - b - the second value
    
    Returns:
        - the eucilean distance between value a and value b
    """
    return np.sqrt(np.sum(np.square(a - b)))

# building the classifier 

class KNN:
    """
    A class to represent the implementation of the KNN (K-nearest neighbours) algorithm.
    
    Attributes:
        - k - the number of nearest neighbours (assumed to be an old value)
        - x_train - the values that will be used to train the classifier
        - x_test - the values that will be used to test the classifier
        
    Methods:
        - get_most_common_element(k_nearest_labels) -> gets the most common element in a list of expected outputs (y_train)
        - predict(unknown_value) -> predicts the nearest neighbour of an unknown_value (a value in x_test)
    """
    
    # initalizing the classifier with k, which is the number of 
    def __init__(self, k, x_train, x_test):
        """
        Constructs all of the necessary attributes for the KNN object.
        
        Params:
            - k - the number of nearest neighbours (assumed to be an old value)
            - x_train - the values that will be used to train the classifier
            - x_test - the values that will be used to test the classifier
        """
        self.k = k
        self.x_train = x_train
        self.x_test = x_test
    
        
    def get_most_common_element(self, k_nearest_labels):
        """
        Gets the most common element in a list of labels.
        
        Params:
            - k_nearest_labels - the expected outputs (or labels) corresponding to the k nearest inputs
            
        Returns:
            - the most common element (either 0 or 1)
        
        """
        
        # initializing tracker variables
        count_0 = 0
        count_1 = 0
        
        # getting the number of occurences for 0 and 1 
        for i in range(len(k_nearest_labels)):
            if k_nearest_labels[i] == 0:
                count_0 += 1
            else:
                count_1 += 1
        
        # returning the class (0 or 1) with the highest number of occurences
        if count_0 > count_1:
            return 0
        else:
            return 1
    
    
    def predict(self, unknown_value):
        """
        Predicts the nearest neighbour of an unknown_value.
        
        Params:
            - unknown_value - the value we are trying to classify (a value in x_test)
        
        Returns:
            - the prediction as a value of either 0 or 1 (the y_test corresponding to our value in x_test)
        """
        
         # extracting all distance values and their corresponding labels (as a list of tuples) between the unknown values (x_test values) and the trained values (x_trainn values).
        k_features = sorted([(eucilean_distance(unknown_value, self.x_train[i]), y_train[i]) for i in range(len(self.x_train))])        
        
        # selecting the three closest distance-label pair (list of tuples)
        k_nearest_features = k_features[:self.k]
        
        # extracting the three labels (as a list of outputs) 
        k_nearest_labels = [tuple[-1] for tuple in k_nearest_features]
       
        # getting and returning the prediction      
        prediction = self.get_most_common_element(k_nearest_labels)
                        
        return prediction

# building the class
knn = KNN(k=3, x_train=x_train, x_test=x_test)

# getting all of the predictions
y_pred = []
for i in range(len(x_test)):
    y_pred.append(knn.predict(x_test[i]))
    
# printing the results
print(classification_report(y_test, y_pred, target_names=['class 0', 'class 1']))



