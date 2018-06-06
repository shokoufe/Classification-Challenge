import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random

from functools import reduce

# fixed random seeds for reproducibality
seed=7
np.random.seed(seed)

#load dataset
dataframe = pd.read_csv("sample.csv", header=None)
print(dataframe.head())

dataset = dataframe.values#[0:900,:]
del dataframe # save memory
print(dataset.shape)
#class labels
classLabels = ['A', 'B', 'C', 'D', 'E']
classData = [ list(filter(lambda line: line[-1] == label, dataset)) for label in classLabels]

# remove bias in the training data
numberOfClassOccurrences = list(map(lambda data: len(data), classData))
print(numberOfClassOccurrences)
lowestNumber = min(numberOfClassOccurrences)
# randomly sample from all classes so that all classes have the same number of instances
dataset_unbiased = list(map(lambda data: [data[i] for i in random.sample(range(len(data)), lowestNumber)], classData))
# concatenate data from classes and mix them again
dataset_unbiased  = reduce(lambda x, y: np.concatenate((x, y)), dataset_unbiased)
np.random.shuffle(dataset_unbiased)

#X = dataset[:,0:295].astype(float)
X = dataset_unbiased[:,0:295].astype(float)
for i in range(X.shape[1]):
    X[:,i] /= np.max(X[:,i]) if np.max(X[:,i]) > 0 else 1
#Y = dataset[:,295]
Y = dataset_unbiased[:,295]

#

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)




# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, dummy_y, test_size = 0.2)







