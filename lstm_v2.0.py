
# coding: utf-8

# In[82]:

import scipy
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


# In[2]:

dataset_filename = 'IanArffDataset.arff'
dataset, meta = arff.loadarff(dataset_filename)

feature_names = meta.names()[:-3]
label_name    = 'binary result'
cat_name      = 'categorized result'

features   = dataset[feature_names]
labels     = dataset[label_name].astype(np.float).reshape((-1, 1))
categories = dataset[cat_name]

addresses = preprocessing.label_binarize(features['address'], classes=[4])

encoder = LabelEncoder()
functions = encoder.fit_transform(features['function'])
functions = np_utils.to_categorical(functions)

responses = dataset['command response'].astype(np.float).reshape((-1, 1))

timestamp_diffs = np.diff(features['time'])
timestamp_diffs = np.insert(timestamp_diffs, 0, 0)


# In[15]:

functions[0]


# In[91]:




# In[3]:

remaining_feature_names = meta.names()[2:14]
remaining_features = dataset[remaining_feature_names]
remaining_features = remaining_features     .view(np.float64)     .reshape(remaining_features.shape + (-1,))



# In[4]:

#NEW - Getting Length from Remaining_features
lengths = remaining_features[:,0].reshape((-1,1))
remaining_features = remaining_features[:,1:len(remaining_features)]
print(remaining_features[1])


# In[8]:

(X_train_rf, Y_train_rf), (X_valid_rf, Y_valid_rf), (X_test_rf, Y_test_rf) = split_dataset(remaining_features, labels, .6 , .2, .2)


# In[76]:

#NEW - Classifying Payload (Remaining features)

IndexesNanPayload = []
IndexesOnlyPressure = []
IndexesNoPressure = []
i = 0
for rf in X_train_rf:
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

assert(len(IndexesNanPayload)+len(IndexesOnlyPressure)+len(IndexesNoPressure)) == X_train_rf.shape[0]

assert(set(IndexesNanPayload)!=set(IndexesOnlyPressure))
assert(set(IndexesOnlyPressure)!=set(IndexesNoPressure))
assert(set(IndexesNoPressure)!=set(IndexesNanPayload))

X_train_rf_OnlyPressure = X_train_rf[IndexesOnlyPressure]
X_train_rf_OnlyPressure = np.array(X_train_rf_OnlyPressure[:,X_train_rf_OnlyPressure.shape[1]-1].reshape((-1,1)))
X_train_rf_NoPressure   = X_train_rf[IndexesNoPressure]
X_train_rf_NoPressure   = np.array(X_train_rf_NoPressure[:,0:X_train_rf_NoPressure.shape[1]-1])
print (X_train_rf_NoPressure)

    


# In[81]:

#Clustering - looking for the best number of K

def clustering(X, samplesize =None, max_k = 10):
    clusters=range(2,max_k)
    samplesize = samplesize
    best_score_k = 0
    
    meandist=[]
    kmeans = KMeans(n_clusters=clusters[0], random_state=0).fit(X)
    labels = kmeans.labels_
    #Average distance measure: calculate the difference of each observation 
    #with all the centroids. Then get which observation is nearest to which centroids,
    #after that add all of them and divide by no of observations. 
    kmeans.predict(X)
    centers = kmeans.cluster_centers_
    print(X.shape)
    meandist.append(sum(np.min(cdist(X, centers, 'euclidean'), axis=1))/ X.shape[0])
    #Calculate silhouette_score (value from [-1,1]) to choose best K--> 1 means the best
    #silhouetteScore = silhouette_score(X_train_rf_OnlyPressure, labels, metric='euclidean', sample_size=samplesize, random_state=0)
    #best_score_k = silhouetteScore
    n_k = 2
    #print (best_score_k)

    for k in range(clusters[1],len(clusters)+clusters[0]):  
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        labels = kmeans.labels_
        kmeans.predict(X)
        centers = kmeans.cluster_centers_
        meandist.append(sum(np.min(cdist(X, centers, 'euclidean'), axis=1))/ X.shape[0])
        #Calculate silhouette_score ([-1,1]) to choose best K--> 1 means the best
        #silhouetteScore = silhouette_score(X_train_rf_OnlyPressure, labels, metric='euclidean', sample_size=samplesize, random_state=0)
#         print(k)
#         print (silhouetteScore)
#         if(silhouetteScore>best_score_k):
#             best_score_k = silhouetteScore
#             n_k = k
#     print (best_score_k)
    #Plot average distance from observations from the cluster centroid 
    #to use the Elbow Method to identify number of clusters to choose
    print(meandist)
    plt.plot(clusters, meandist)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average distance')
    plt.title('Selecting k with the Elbow Method')
    plt.show()
    return n_k, best_score_k
k, bestscore = clustering(X_train_rf_OnlyPressure)
k, bestscore = clustering(X_train_rf_NoPressure,X_train_rf_NoPressure.shape[0],10)


# In[ ]:

#Create one hot encoding per cluster (total amount of clusters = 10)


# In[14]:

new_features = np.column_stack((
    addresses,
    functions,
    responses,
    timestamp_diffs,
    lengths,
    remaining_features
))


# In[29]:

new_features[0]


# In[18]:

get_ipython().magic('matplotlib inline')
def make_histograms(features, feature_names):
    n_bins = 10
    for n in range(features.shape[1]):
        xs = features[:, n]
        xs = xs[~np.isnan(xs)]
        plt.hist(xs, n_bins, normed=1, histtype='bar')
        plt.axvline(xs.mean(), color='b', linestyle='dashed', linewidth=2)
        plt.title(feature_names[n])
        plt.show()

make_histograms(remaining_features, remaining_feature_names)  


# In[7]:

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


# In[66]:

(X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = split_dataset(new_features, labels, .1 , .1, .8)


# In[67]:

range(X_train.shape[1] - 11, X_train.shape[1])


# In[65]:

# #Divide dataset depending on Payload features (Nan features, Only pressure, All payload features but pressure)
# IndexesNanPayload = []
# IndexesOnlyPressure = []
# IndexesNoPressure = []
# i = 0
# for sample in X_train:
#     payload = sample[X_train.shape[1] - 1: X_train.shape[1]]
#     Nan = all(str(i) == 'nan' for i in payload)
#     #print (Nan)
#     if (Nan):
#         IndexesNanPayload.append(X_train[i])
#     i += 1
# print(IndexesNanPayload)


# In[52]:

IndexesNanPayload


# In[22]:

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

def preprocess_data(Xs, train_data_stats = None):
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
    
X_train_preprocessed, X_train_stats = preprocess_data(X_train)
X_val_preprocessed,  _ = preprocess_data(X_valid, X_train_stats)
X_test_preprocessed, _ = preprocess_data(X_test, X_train_stats)
X_train_preprocessed.shape


# In[23]:

def make_sequences(Xs, Ys, maxlen):
    X_seq, Y_seq = [], []
    for i in range(0, Xs.shape[0] - maxlen):
        X_seq.append(Xs[i: i+maxlen])
        Y_seq.append(Ys[i: i+maxlen])
    return np.array(X_seq), np.array(Y_seq)

maxlen = 7
X_train_seq, Y_train_seq = make_sequences(X_train_preprocessed, Y_train, maxlen)
X_val_seq, Y_val_seq     = make_sequences(X_val_preprocessed,   Y_valid, maxlen)
X_test_seq, Y_test_seq   = make_sequences(X_test_preprocessed,  Y_test, maxlen)
X_train_seq.shape, Y_train_seq.shape


# In[ ]:




# In[46]:

import importlib, Algorithm
Algorithm = importlib.reload(Algorithm)

model = Algorithm.one_layer_lstm(maxlen)

iters = 1

for iteration in range(60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train_seq, Y_train_seq, batch_size=128, epochs=1)
    score = model.evaluate(X_val_seq, Y_val_seq, batch_size=128, verbose=1)
    print(score)


# In[51]:

score = model.evaluate(X_test_batched, Y_test_batched, batch_size=100, verbose=1)


# In[27]:

predicted = model.predict(X_test_batched, batch_size=100, verbose=1)



# In[40]:

predicted == Y_test_batched.astype(np.float).reshape((-1, 1))


# In[41]:

labels


# In[53]:

Y_train_batched


# In[ ]:



