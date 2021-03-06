#!/usr/bin/env python3
import scipy
import Algorithm
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

new_features = np.column_stack((addresses, lengths, crcs, timestamp_diffs, packet_inspected_features))
#print "Length: "+str(len(new_features))
#Split the data - Training set and testing set
X_train = new_features[:164777,:]
X_crossValidation = new_features[164778:219704,:]
X_test = new_features[219705:len(new_features)-1,:]
#print(new_features)
#print X_train
#print str(len(X_train))
#print X_crossValidation
#print str(len(X_crossValidation))
#print X_test
#print str(len(X_test))

#Preprocess - Mean removal and variance scaling
X_train_scaled = preprocessing.scale(X_train)

scaler = preprocessing.StandardScaler().fit(X_train)
X_crossValidation_scaled = scaler.transform(X_crossValidation)
X_test_scaled = scaler.transform(X_test)         
# After preprocessing, function feature is added (since this one didn't need to be preprocessed )
X_train_scaled = np.column_stack((X_train_scaled, functions[:164777]))
X_crossValidation_scaled = np.column_stack((X_crossValidation_scaled,functions[164778:219704]))
X_test_scaled = np.column_stack((X_test_scaled,functions[219705:len(new_features)-1]))

y_train = labels[:164777]
y_crossVal = labels[164778:219704]
y_testData = labels[219705:len(new_features)-1]

#Take only few samples
X_train_scaled = X_train_scaled[:100]
y_train = y_train[:100]
X_crossValidation_scaled = X_crossValidation_scaled[101:151]
y_crossVal = y_crossVal[101:151]

sequence_length = 7
X_train = []
Y_train = []
for index in range(len(X_train_scaled) - sequence_length):
    X_train.append(X_train_scaled[index: index + sequence_length])
    Y_train.append(y_train[index + sequence_length])
X_train = np.array(X_train) 
Y_train = np.array(Y_train)
Y_train = Y_train[:len(X_train)]
X_crossValidate = []
Y_crossVal = []
for index in range(len(X_crossValidation_scaled) - sequence_length):
    X_crossValidate.append(X_crossValidation_scaled[index: index + sequence_length])
    Y_crossVal.append(y_crossVal[index + sequence_length])
X_crossValidate = np.array(X_crossValidate) 
Y_crossVal = np.array(Y_crossVal)
Y_crossVal = Y_crossVal[:len(X_crossValidate)]

#Convert arrays in (#examples, #values in sequences, dim. of each value)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 16))
#X_train = X_train_scaled.reshape((-1,X_train_scaled.shape[0],X_train_scaled.shape[1]))
X_crossValidate = np.reshape(X_crossValidate, (X_crossValidate.shape[0], X_crossValidate.shape[1],16))

Algorithm.lstm(X_train, Y_train, X_crossValidate, Y_crossVal)


