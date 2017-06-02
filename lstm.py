#!/usr/bin/env python3
import scipy
import numpy as np
from scipy.io import arff
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

dataset_filename = 'IanArffDataset.arff'
dataset, meta = arff.loadarff(dataset_filename)

feature_names = meta.names()[:-3]
label_name = 'binary result'
cat_name   = 'categorized result'

features = dataset[feature_names]
labels   = dataset[label_name]
categories = dataset[cat_name]

#
# If address is not 4, then it's always DoS attack in our dataset
#
# address == 4 -> 1
# address != 4 -> 0
#
addresses = preprocessing.label_binarize(features['address'], classes=[4])

# Scale function code feature to be between 0 to 1
functions = features['function']
# functions_unique = np.unique(functions)
# for i in range(len(functions)-1):
#     nfunction = functions[i]
#     index = np.where(functions_unique == nfunction)[0][0]
#     functions[i] = index

encoder = LabelEncoder()
encoder.fit(functions)
feature_labels = encoder.transform(functions)
functions_encoder = np_utils.to_categorical(functions)

#min_max_scaler = preprocessing.MinMaxScaler()
#functions = min_max_scaler.fit_transform(features['function'].reshape(-1, 1))

# Normalize packet lengths and CRC to unit gaussian
# Subtract mean and divide by standard deviation
#lengths = preprocessing.scale(features['length']).reshape(-1, 1)
#crcs = preprocessing.scale(features['crc rate']).reshape(-1, 1)
lengths = features['length']
crcs = features['crc rate']
# Compute differences between consequitive packets and normalize to gaussian
timestamp_diffs = np.diff(features['time'])
timestamp_diffs = np.insert(timestamp_diffs, 0, 0)
#timestamp_diffs = preprocessing.scale(timestamp_diffs).reshape(-1, 1)

# Select features returned by deep packet inspection of the modbus frames
# We have to convert from structured numpy arrays to standard numpy arrays
# Lastly, impute missing values by replacing them with mean

# could be 'median' or 'most_frequenty'
impute_strategy = 'mean'

packet_inspected_feature_names = meta.names()[3:14]
packet_inspected_features = dataset[packet_inspected_feature_names]
packet_inspected_features = packet_inspected_features \
	.view(np.float64) \
	.reshape(packet_inspected_features.shape + (-1,))
imp = preprocessing.Imputer(missing_values='NaN', strategy=impute_strategy, axis=0)
packet_inspected_features = imp.fit_transform(packet_inspected_features)

new_features = np.column_stack((addresses, functions, lengths, crcs, timestamp_diffs, packet_inspected_features))
print "Length: "+str(len(new_features))
#Split the data - Training set and testing set
X_train = new_features[:164777,:]
X_crossValidation = new_features[164778:219704,:]
X_test = new_features[219705:len(new_features)-1,:]
print(new_features)
#print X_train
print str(len(X_train))
#print X_crossValidation
print str(len(X_crossValidation))
#print X_test
print str(len(X_test))

#Preprocess - Mean removal and variance scaling
X_train_scaled = preprocessing.scale(X_train)
print X_train

scaler = preprocessing.StandardScaler().fit(X_train)
X_crossValidation_scaled = scaler.transform(X_crossValidation)
X_test_scaled = scaler.transform(X_test)         


