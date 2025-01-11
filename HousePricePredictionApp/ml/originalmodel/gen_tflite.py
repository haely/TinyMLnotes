from sklearn.preprocessing import LabelEncoder
import sklearn.preprocessing as skpp
import scipy.stats as stats
from scipy.special import boxcox1p
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import re

from keras.models import Sequential
from keras.layers import BatchNormalization 
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint #we can control our model if going well during validation part or not
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import Adam
import tensorflow as tf

# --- Define your model architecture ---

model = Sequential()

model.add(Dense(units=256, activation='linear', input_dim=17))  # Set input_dim
model.add(BatchNormalization())
model.add(Dense(units=128, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=64, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=32, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=16, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=8, activation='linear'))
model.add(BatchNormalization())
model.add(Dense(units=1, activation='linear'))

# --- Compile the model ---

model.compile(loss='mean_squared_logarithmic_error',
               optimizer='adam') 


# --- Convert to TensorFlow Lite ---

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# --- Save the .tflite file ---

with open('house_price_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model saved to home_price_model.tflite")