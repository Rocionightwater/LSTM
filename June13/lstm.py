
# coding: utf-8

# In[65]:

import scipy
import math
import numpy as np
from scipy.io import arff
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.preprocessing import OneHotEncoder
import sklearn.preprocessing


# In[66]:

dataset_filename = 'IanArffDataset.arff'
dataset, meta = arff.loadarff(dataset_filename)

feature_names = meta.names()[:-3]
label_name    = 'binary result'
cat_name      = 'categorized result'

features   = dataset[feature_names]
labels     = dataset[label_name].astype(np.float).reshape((-1, 1))
categories = dataset[cat_name].astype(np.float)

cat_encoder = LabelEncoder()
categories = cat_encoder.fit_transform(categories)
categories = np_utils.to_categorical(categories)

addresses = preprocessing.label_binarize(features['address'], classes=[4])

encoder = LabelEncoder()
functions = encoder.fit_transform(features['function'])
functions = np_utils.to_categorical(functions)

responses = dataset['command response'].astype(np.float).reshape((-1, 1))

timestamp_diffs = np.diff(features['time'])
timestamp_diffs = np.insert(timestamp_diffs, 0, 0)


# In[67]:

remaining_feature_names = meta.names()[2:14]
remaining_features = dataset[remaining_feature_names]
remaining_features = remaining_features     .view(np.float64)     .reshape(remaining_features.shape + (-1,))


# In[68]:

#NEW - Getting Length from Remaining_features
lengths = remaining_features[:,0].reshape((-1,1))
remaining_features = remaining_features[:,1:]
print(remaining_features[1])


# In[69]:

def split_dataset(data, labels, train_per_split, val_per_split, test_per_split):
    train_percentage_split      = train_per_split
    validation_percentage_split = val_per_split
    test_percentage_split       = test_per_split
    assert (train_percentage_split + validation_percentage_split + test_percentage_split) == 1.0
    
    train_end_index = int(data.shape[0] * train_percentage_split)
    valid_end_index = int(data.shape[0] * validation_percentage_split) + train_end_index

    X_train, Y_train = data[:train_end_index], labels[:train_end_index]
    X_valid, Y_valid = data[train_end_index:valid_end_index], labels[train_end_index:valid_end_index]
    X_test,  Y_test  = data[valid_end_index:], labels[valid_end_index:]
    assert X_train.shape[0] + X_valid.shape[0] + X_test.shape[0] == data.shape[0]
    
    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)


# In[70]:

#Spliting data - binary labels
(X_train_rf, Y_train_rf), (X_valid_rf, Y_valid_rf), (X_test_rf, Y_test_rf) = split_dataset(remaining_features, labels, .6 , .2, .2)


# In[71]:

#Sliting data - categorical labels
(_, Y_train_cat), (_, Y_valid_cat), (_, Y_test_cat) = split_dataset(remaining_features, categories, .6 , .2, .2)


# In[72]:

#NEW - Classifying Payload (Remaining features)
def ClassifyPayload(X_rf,Y_rf):
    IndexesNanPayload = []
    IndexesOnlyPressure = []
    IndexesNoPressure = []
    i = 0
    for rf in X_rf:
        Nan = all(str(i) == 'nan' for i in rf[len(rf)-2:len(rf)])
        OnlyPressure = str(rf[len(rf)-2])=='nan' and str(len(rf)!='nan')
    #     #print (Nan)
        if (Nan):
            IndexesNanPayload.append(i)
        elif (OnlyPressure):
            IndexesOnlyPressure.append(i)
        else:
            IndexesNoPressure.append(i)
        i += 1
    #print(IndexesNoPressure)

    assert(len(IndexesNanPayload)+len(IndexesOnlyPressure)+len(IndexesNoPressure)) == X_rf.shape[0]

    assert(set(IndexesNanPayload)!=set(IndexesOnlyPressure))
    assert(set(IndexesOnlyPressure)!=set(IndexesNoPressure))
    assert(set(IndexesNoPressure)!=set(IndexesNanPayload))

    X_rf_OnlyPressure = X_rf[IndexesOnlyPressure]
    Y_rf_OnlyPressure = Y_rf[IndexesOnlyPressure]
    X_rf_OnlyPressure = np.array(X_rf_OnlyPressure[:,X_rf_OnlyPressure.shape[1]-1].reshape((-1,1)))
    X_rf_NoPressure   = X_rf[IndexesNoPressure]
    Y_rf_NoPressure   = Y_rf[IndexesNoPressure]
    X_rf_NoPressure   = np.array(X_rf_NoPressure[:,0:X_rf_NoPressure.shape[1]-1])

    # IndexesNanPayload   = [str(x) for x in IndexesNanPayload]
    # IndexesOnlyPressure = [str(x) for x in IndexesOnlyPressure]
    # IndexesNoPressure   = [str(x) for x in IndexesNoPressure]
    #print (Y_rf_OnlyPressure)
    #print(IndexesNoPressure)
    return X_rf_OnlyPressure,X_rf_NoPressure,Y_rf_OnlyPressure,Y_rf_NoPressure,IndexesNanPayload,IndexesOnlyPressure,IndexesNoPressure


# In[73]:

#Clustering - looking for the best number of K

def clustering_getK(X, max_k = 10):
    clusters=range(2,max_k)
    best_score_k = 0
    
    meandist=[]
    kmeans = KMeans(n_clusters=clusters[0], random_state=0).fit(X)
    labels = kmeans.labels_
    #Average distance measure: calculate the difference of each observation 
    #with all the centroids. Then get which observation is nearest to which centroids,
    #after that add all of them and divide by no of observations. 
    predictions = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    print(X.shape)
    meandist.append(sum(np.min(cdist(X, centers, 'euclidean'), axis=1))/ X.shape[0])
    n_k = 2
    #print (best_score_k)

    for k in range(clusters[1],len(clusters)+clusters[0]):  
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        labels = kmeans.labels_
        predictions = kmeans.predict(X)
        centers = kmeans.cluster_centers_
        meandist.append(sum(np.min(cdist(X, centers, 'euclidean'), axis=1))/ X.shape[0])
    print(meandist)
    plt.plot(clusters, meandist)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average distance')
    plt.title('Selecting k with the Elbow Method')
    plt.show()
    return n_k, best_score_k


# In[74]:

#Once we know the best number of clusters (k) we perform the clustering
def clustering(X, k, indexes):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=100).fit(X)
    labels = kmeans.labels_
    print(set(labels))
    #Average distance measure: calculate the difference of each observation 
    #with all the centroids. Then get which observation is nearest to which centroids,
    #after that add all of them and divide by no of observations. 
    predictions = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    print(centers)
    return [(a,b) for a,b in zip(indexes, predictions)]


# In[75]:

from sklearn.mixture import GaussianMixture

def gmm(X, Y_train, X_val,Y_val, k, cov_type = "full", strategy = "None"):
    #If the strategy we want is supervised
    if strategy == "S":
        #If the covariance type we want is a specific one (by default it is "full")
        if cov_type != "None":
            estimator = GaussianMixture(n_components=k, init_params='random', covariance_type=cov_type,
                                                         max_iter=200, random_state=0)

            n_classes = len(np.unique(Y_train_rf))
            estimator.means_init = np.array([X[Y_train.reshape(-1) == i].mean(axis=0)
                                           for i in range(n_classes)])
            estimator.fit(X)
            train_pred = estimator.predict(X)
            train_accuracy = np.mean(train_pred.ravel() == Y_train.ravel()) * 100

            val_pred = estimator.predict(X_val)
            test_accuracy = np.mean(val_pred.ravel() == Y_val.ravel()) * 100
            name = cov_type
            printGMM(name, train_accuracy,test_accuracy,train_pred,val_pred)

        else:
            # Try GMMs using different types of covariances.
            estimators = dict((cov_type, GaussianMixture(n_components=k, init_params='random', covariance_type=cov_type,
                                                         max_iter=200, random_state=0))
                              for cov_type in ['spherical', 'diag', 'tied', 'full'])
            n_classes = len(np.unique(Y_train_rf))
            print(n_classes)

            for index, (name, estimator) in enumerate(estimators.items()):
                # Since we have class labels for the training data, we can
                # initialize the GMM parameters in a supervised manner.


                estimator.means_init = np.array([X[Y_train.reshape(-1) == i].mean(axis=0)
                                           for i in range(n_classes)])

                estimator.fit(X)
                train_pred = estimator.predict(X)
                train_accuracy = np.mean(train_pred.ravel() == Y_train.ravel()) * 100

                val_pred = estimator.predict(X_val)
                test_accuracy = np.mean(val_pred.ravel() == Y_val.ravel()) * 100
                printGMM(name, train_accuracy,test_accuracy,train_pred,val_pred)
        
    else:
        aic_list = []
        clusters = range(2,k+1)
        aic_value = float("inf")
        best_k = 0
        for k1 in range(2,k+1):
            estimator = GaussianMixture(n_components=k1, init_params='random', covariance_type=cov_type,
                                                                 max_iter=200, random_state=0)
            
            estimator = estimator.fit(X)
            aic = estimator.aic(X_val)
            aic_list.append(aic)
            if aic <= aic_value:
                best_k = k1
                aic_value = aic
        estimator = GaussianMixture(n_components=k, init_params='random', covariance_type=cov_type,
                                                                 max_iter=200, random_state=0)
        estimator = estimator.fit(X)
        aic = estimator.aic(X_val)
        print(best_k)
        print (aic)
        print(estimator.means_)
        train_pred = estimator.predict(X)
        val_pred = estimator.predict(X_val)
        name = cov_type
        train_accuracy = "null"
        test_accuracy  = "null"
        printGMM(name, train_accuracy,test_accuracy,train_pred,val_pred)
        
    plt.plot(clusters, aic_list)
    plt.xlabel('Number of clusters')
    plt.ylabel('AIC value')
    plt.title('Selecting k with the AIC Method')
    plt.show()
    return train_pred, val_pred, best_k

def printGMM(name, train_accuracy,test_accuracy,train_pred,val_pred):
    
    print(name+" Accuray for training set = "+str(train_accuracy))
    print(name+" Accuracy for validation set = "+str(test_accuracy))
    print(name+" Predictions for training set = "+str(train_pred))
    print(name+" Predictions for validation set = "+str(val_pred)+"\n")


# In[76]:

def defineClusters(train_set,val_set,test_set, i):
    train_set = train_set + i
    val_set   = val_set   + i
    test_set  = test_set  + i
    return train_set, val_set, test_set


# In[77]:

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def impute_simple(Xs, strategy = 'mean', stats = None):
    if not stats:
        Xs_imputer = preprocessing.Imputer(missing_values='NaN', strategy=strategy, axis=0)
        Xs  = Xs_imputer.fit_transform(Xs)
        Xs_stats = {}
        Xs_stats['impute'] = Xs_imputer.statistics_
        return Xs, Xs_stats
    else:
        for f_index in range(Xs.shape[1]):
            feature = Xs[:, f_index]
            feature[np.isnan(feature)] = stats['impute'][f_index]
            Xs[:, f_index] = feature
        return Xs, None

def normalize(Xs, stats = None):
    if not stats:
        Xs_scaler = preprocessing.StandardScaler()
        Xs = Xs_scaler.fit_transform(Xs)
        Xs_stats = {
            'mean'  : Xs_scaler.mean_,
            'scale' : Xs_scaler.scale_
        }
        return Xs, Xs_stats
    else:
        for f_index in range(Xs.shape[1]):
            feature = Xs[:, f_index]
            feature -= stats['mean'][f_index]
            feature /= stats['scale'][f_index]
            Xs[:, f_index] = feature
        return Xs, None

def preprocess_data(Xs, strategy,Ys=None, train_data_stats = None, max_k =None, indexes = None, X_val= None,Y_val = None, cov_type = None, supervised = None):
    
    if strategy == "mean":
        #split again to pre-process each column adequately
        #no need to further pre-process these features
        addresses = Xs[:, 0]
        functions = Xs[:, 1:29]
        responses = Xs[:, 29]

        # features to be further pre-processed
        time_diffs   = Xs[:, 30].reshape((-1, 1))
        payload_feas = Xs[:, 31:-1]
        pressures    = Xs[:, -1].reshape((-1, 1))

        stats = None
        if not train_data_stats:
            stats = {}
            time_diffs, stats['time_diffs'] = normalize(time_diffs)

            payload_feas, payload_feas_impute_stats = impute_simple(payload_feas)
            payload_feas, payload_feas_norm_stats   = normalize(payload_feas)
            stats['payload'] = merge_two_dicts(payload_feas_impute_stats, payload_feas_norm_stats)

            pressures, pressures_impute_stats = impute_simple(pressures)
            pressures, pressures_norm_stats   = normalize(pressures)
            stats['pressures'] = merge_two_dicts(pressures_impute_stats, pressures_norm_stats)

        else:
            time_diffs, _ = normalize(time_diffs, train_data_stats['time_diffs'])

            payload_feas, _ = impute_simple(payload_feas, stats = train_data_stats['payload'])
            payload_feas, _ = normalize(payload_feas, train_data_stats['payload'])

            pressures, _ = impute_simple(pressures, strategy = 'median', stats = train_data_stats['pressures'])
            pressures, _ = normalize(pressures, train_data_stats['pressures'])


        Xs_preprocessed = np.column_stack((
            addresses,
            functions,
            responses,
            time_diffs,
            payload_feas,
            pressures
        ))

        return Xs_preprocessed, stats
    elif strategy == "clustering":
        X_OnlyPressure,X_NoPressure,Y_OnlyPressure,Y_NoPressure,IndexesNan,IndexesOP,IndexesNP = ClassifyPayload(Xs,Ys)
        return X_OnlyPressure,X_NoPressure,Y_OnlyPressure,Y_NoPressure,IndexesNan,IndexesOP,IndexesNP
    elif strategy == "kmeansElbow":
        clustering_getK(Xs,max_k)
    elif strategy == "kmeans":
        return clustering(Xs,max_k,indexes) 
    elif strategy == "gmm":
        X_pred, val_pred, n_clust = gmm(Xs, Ys, X_val, Y_val, max_k, cov_type, supervised)
        return X_pred, val_pred, n_clust
        
        
        
        
#---------------------------------Perform mean stratrgy--------------------------------------# 
# X_train_preprocessed, X_train_stats = preprocess_data(X_train,"mean")
# X_val_preprocessed,  _ = preprocess_data(X_valid, X_train_stats"mean")
# X_test_preprocessed, _ = preprocess_data(X_test, X_train_stats"mean")

X_train_rf_OnlyPressure,X_train_rf_NoPressure,Y_train_rf_OnlyPressure,Y_train_rf_NoPressure,IndexesNan_train,IndexesOP_train,IndexesNP_train = preprocess_data(X_train_rf,"clustering",Y_train_rf)
X_val_rf_OnlyPressure,X_val_rf_NoPressure, Y_val_rf_OnlyPressure,Y_val_rf_NoPressure,IndexesNan_val,IndexesOP_val,IndexesNP_val  = preprocess_data(X_valid_rf,"clustering",Y_valid_rf)
X_test_rf_OnlyPressure, X_test_rf_NoPressure, Y_test_rf_OnlyPressure, Y_test_rf_NoPressure,IndexesNan_test,IndexesOP_test,IndexesNP_test = preprocess_data(X_test_rf, "clustering", Y_test_rf)
#----------------------------Perform Kmeans strategy - look for the best k--------------------#
#preprocess_data(X_train_rf_OnlyPressure,"kmeansElbow",None,None,10)
#preprocess_data(X_train_rf_NoPressure, "kmeansElbow",None,None,10)
#------------------------From Elbow I got K=5 as the best - OnlyPressure Payload---------------#
#dictOP_Kmeans = preprocess_data(X_train_rf_OnlyPressure,"kmeans",None,None,5,IndexesOP_train)   
#-----------------------From Elbow I got K=4 as the best - NoPressure Payload-------------------#
#dictNP_Kmeans = preprocess_data(X_train_rf_NoPressure,"kmeans",None,None,4,IndexesNP_train) 
#------------------------------Perform GMM------------------------------------------------------#
train_pred_OnlyPress, val_pred_OnlyPress, L = preprocess_data(X_train_rf_OnlyPressure, "gmm", Y_train_rf_OnlyPressure,None, 7, None, X_val_rf_OnlyPressure, Y_val_rf_OnlyPressure, "full", "None")
train_pred_NoPress,   val_pred_NoPress,   K = preprocess_data(X_train_rf_NoPressure,"gmm", Y_train_rf_NoPressure,None, 8, None, X_val_rf_NoPressure, Y_val_rf_NoPressure, "full", "None")
tr, test_pred_OnlyPress, TL = preprocess_data(X_train_rf_OnlyPressure,"gmm", Y_train_rf_OnlyPressure, None, 7, None, X_test_rf_OnlyPressure,Y_test_rf_OnlyPressure, "full",  "None" )
tr, test_pred_NoPress ,  TK = preprocess_data(X_train_rf_NoPressure, "gmm", Y_train_rf_NoPressure,None, 8, None, X_test_rf_NoPressure, Y_test_rf_NoPressure, "full", "None")

train_OnlyPress_Clusters, val_OnlyPress_Clusters, test_OnlyPress_Clusters = defineClusters(train_pred_OnlyPress,val_pred_OnlyPress,test_pred_OnlyPress,1)
train_NoPress_Clusters, val_NoPress_Clusters, test_NoPress_Clusters = defineClusters(train_pred_NoPress  ,val_pred_NoPress,test_pred_NoPress,  L+1)

#THIS IS ONLY PRINTING TO CHECK ALL GOES WELL
print(train_OnlyPress_Clusters)

n_classes = len(np.unique(train_OnlyPress_Clusters))
print(n_classes)

print(val_OnlyPress_Clusters)

n_classes = len(np.unique(val_OnlyPress_Clusters))
print(n_classes)

print(test_OnlyPress_Clusters)

n_classes = len(np.unique(test_OnlyPress_Clusters))
print(n_classes)
print(train_NoPress_Clusters)
n_classes = len(np.unique(train_NoPress_Clusters))
print(n_classes)
print(val_NoPress_Clusters)
n_classes = len(np.unique(val_NoPress_Clusters))
print(n_classes)
print(test_NoPress_Clusters)
n_classes = len(np.unique(test_NoPress_Clusters))
print(n_classes)

#One hot encoding (7 clusters)
label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(L+K+1))

train_OP_Encoder = label_binarizer.transform(train_OnlyPress_Clusters)
val_OP_Encoder   = label_binarizer.transform(val_OnlyPress_Clusters)
test_OP_Encoder  = label_binarizer.transform(test_OnlyPress_Clusters)

train_NP_Encoder = label_binarizer.transform(train_NoPress_Clusters)
val_NP_Encoder   = label_binarizer.transform(val_NoPress_Clusters)
test_NP_Encoder  = label_binarizer.transform(test_NoPress_Clusters)

pred_nan_train = np.zeros(len(IndexesNan_train))
pred_nan_val = np.zeros(len(IndexesNan_val))
pred_nan_test = np.zeros(len(IndexesNan_test))

train_Nan_Encoder = label_binarizer.transform(pred_nan_train)
val_Nan_Encoder   = label_binarizer.transform(pred_nan_val)
test_Nan_Encoder  = label_binarizer.transform(pred_nan_test)


# In[78]:

indexes_train = IndexesOP_train + IndexesNP_train + IndexesNan_train

encoders_train = np.concatenate((train_OP_Encoder,train_NP_Encoder,train_Nan_Encoder))
payload_trainSet = sorted(zip(indexes_train,encoders_train), key=lambda x: x[0])
payload_trainSet = np.array([x[1] for x in payload_trainSet])

indexes_val  = IndexesOP_val + IndexesNP_val + IndexesNan_val
encoders_val = np.concatenate((val_OP_Encoder,val_NP_Encoder,val_Nan_Encoder))
payload_valSet = sorted(zip(indexes_val,encoders_val), key=lambda x: x[0])
payload_valSet = np.array([x[1] for x in payload_valSet])


indexes_test = IndexesOP_test + IndexesNP_test + IndexesNan_test
encoders_test = np.concatenate((test_OP_Encoder,test_NP_Encoder,test_Nan_Encoder))
payload_testSet = sorted(zip(indexes_test,encoders_test), key=lambda x: x[0])
payload_testSet = np.array([x[1] for x in payload_testSet])


# In[79]:

#Join all the features for final training, validation and testing sets
def createFinalSets(start,end,X_rf):
    new_features = np.column_stack((
    addresses[start:end],
    functions[start:end],
    responses[start:end],
    timestamp_diffs[start:end],
    lengths[start:end],
    X_rf
    ))
    return new_features

train_end = len(payload_trainSet)
valid_end = train_end + len(payload_valSet)
test_end  = valid_end + len(payload_testSet)
    
trainSet_features = createFinalSets(0,train_end, payload_trainSet)
validSet_features = createFinalSets(train_end, valid_end, payload_valSet)
testSet_features  = createFinalSets(valid_end, test_end, payload_testSet)


# In[80]:

assert trainSet_features.shape[0] + validSet_features.shape[0] + testSet_features.shape[0] == dataset.shape[0]


# In[81]:

def make_sequences(Xs, Ys, maxlen):
    X_seq, Y_seq = [], []
    for i in range(0, Xs.shape[0] - maxlen):
        X_seq.append(Xs[i: i+maxlen])
        Y_seq.append(Ys[i+1: i+maxlen+1])
    return np.array(X_seq), np.array(Y_seq)

maxlen = 50
X_train_seq, Y_train_seq = make_sequences(trainSet_features, Y_train_cat, maxlen)
X_val_seq,   Y_val_seq   = make_sequences(validSet_features, Y_valid_cat, maxlen)
X_test_seq,  Y_test_seq  = make_sequences(testSet_features,  Y_test_cat,  maxlen)
X_train_seq.shape, Y_train_seq.shape


# In[ ]:




# In[83]:

import importlib, Algorithm
Algorithm = importlib.reload(Algorithm)
#Parameters
inp  = X_train_seq.shape[2]
outp = Y_train_seq.shape[2]
hidden = 128
model = Algorithm.one_layer_lstm(maxlen,inp,hidden,outp)

iters = 1

for iteration in range(iters):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train_seq, Y_train_seq, batch_size=128, epochs=1)
    loss, metrics = model.evaluate(X_val_seq, Y_val_seq, batch_size=128, verbose=1)
    print(loss)


# In[ ]:

test_loss, test_metrics = model.evaluate(X_test_seq, Y_test_seq, batch_size=128, verbose=1)


# In[ ]:

predictions = model.predict(X_train_seq, batch_size=128, verbose=1)
predictions = set(np.argmax(predictions[:2], axis=2))
print(predictions)

predictions = model.predict(X_val_seq, batch_size=128, verbose=1)
predictions = set(np.argmax(predictions[:2], axis=2))
print(predictions)

predictions = model.predict(X_test_seq, batch_size=128, verbose=1)
predictions = set(np.argmax(predictions[:2], axis=2))
print(predictions)


# In[ ]:

set(np.argmax(predictions, axis=2).reshape(-1))


# In[ ]:

print(predictions[:2])
np.argmax(predictions[:2], axis=1)


# In[ ]:




# In[ ]:



