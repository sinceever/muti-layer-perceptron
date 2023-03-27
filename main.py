import time
import pandas as pd
import numpy as np
import umap
import sklearn.metrics as metrique
from pandas import Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from collections import Counter
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import LSTM, Dense, Embedding, Dropout, Input, Attention, Layer, Concatenate, Permute, Dot, Multiply, \
    Flatten
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.models import Sequential
from keras import backend as K, regularizers, Model, metrics
from keras.backend import cast
from imblearn.over_sampling import SMOTE


# using swarm intelligence algorithm to reduce dimensionality
data = pd.read_csv('./creditcard.csv', na_filter=True)
col_del = ['Time', 'V5', 'V6', 'V7', 'V8', 'V9', 'V13', 'V15', 'V16', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24',
           'V25', 'V26', 'V27', 'V28', 'Amount']

tr_data = data.drop(col_del, axis=1)
print('Size after swarm:', tr_data.shape)

X = tr_data.drop(['Class'], axis='columns')
Label_Data = tr_data['Class']

# Generate and plot imbalanced classification dataset
# summarize class distribution
counter = Counter(tr_data['Class'])
print(counter)

# scatter plot of examples by class label
# for label, _ in counter.items():
#     row_ix = np.where(tr_data['Class'] == label)[0]


# Standardizing the data
X_s = StandardScaler().fit_transform(X)
# Use UMAP to project data into 3-dimensional space
start = time.perf_counter()  # tick
print('UMAPing starts at: ', time.ctime())
reducer = umap.UMAP(n_components=3, low_memory=False)
X_embedding = reducer.fit_transform(X_s)
print('Data size after UMAP', X_embedding.shape)
end = time.perf_counter()
print('UMAPing ends after: {:.2f} minutes'.format((end - start)/60))  # tock


# Use SMOTE to transform the dataset
# generate new data points to make a balanced dataset
X_r = pd.DataFrame(X_embedding)
oversample = SMOTE()
X_r2, y = oversample.fit_resample(X_r, Label_Data)
# summarize the new class distribution
counter = Counter(y)
print(counter)

# scatter plot of examples by class label
# for label, _ in counter.items():
#     row_ix = np.where(y == label)[0]


# Save preprocessed data
# add labels to data
X_r3 = np.c_[X_r2, y]
# X_r3 = np.c_[X_r2, y]
np.savetxt(r'C:\Laurentian\Winter 2023\Machine Learning\Project\umap_creditcard.csv', X_r3, delimiter=',', fmt='%f')


# Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X_r2, y, test_size=0.3)
# X_train.shape
# X_test.shape
