import os
import pywt
import time
import random
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.svm import SVC
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.utils import resample
from keras.models import Sequential
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from keras.layers import Dense, Activation
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout,MaxPooling1D,GlobalAveragePooling1D

tf.random.set_seed(42)
np.random.seed(42)


# import MIT-BIH Arrhythmia Data
data_train = pd.read_csv('../content/drive/MyDrive/MITBIH/mitbih_train.csv',header=None)
data_test = pd.read_csv('../content/drive/MyDrive/MITBIH/mitbih_test.csv',header=None)

# check class labels of BIH dataset
print("BIH train class label: " + str(list(data_train[data_train.columns[-1]].unique())))
print("BIH test class label: " + str(list(data_test[data_test.columns[-1]].unique())))

# shuffle the DataFrame rows
data_train = data_train.sample(frac = 1)
data_test = data_test.sample(frac = 1)

#combining two data into one
df = pd.concat([data_train, data_test], axis=0).sample(frac=1.0, random_state =0).reset_index(drop=True)

df.shape

df.info()

df.columns

df[187].unique()

df.dtypes

#looking at missing values for each column
df.isna().sum()

#missing values for entire dataset
df.isna().sum().sum()

#plot graphs of normal and abnormal ECG to visualise the trends
Normal_Beat = df[df.loc[:,140] ==0][:1]
Supraventricular_Beat = df[df.loc[:,140] ==1][:1]
Premature_ventricular_contraction_Beat = df[df.loc[:,140] ==2][:1]
Fusion_Beat = df[df.loc[:,140] ==3][:1]
Unclassifiable_Beat = df[df.loc[:,140] ==4][:1]

# split the data into labels and features
ecg_data = df.iloc[:,:-1]
labels = df.iloc[:,-1]

# Plot original ECG signal
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(ecg_data.shape[0]), y=ecg_data[30], mode='lines', name='Original ECG signal'))
fig.show()





























