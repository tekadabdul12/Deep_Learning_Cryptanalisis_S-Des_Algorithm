import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler




df =pd.read_csv('diabetes.csv')
#print(df.head())

xtrain = df.values
xtrain = np.delete(xtrain,8,axis=1)

#print(xtrain)

xtrain = MinMaxScaler().fit_transform(xtrain)


ytrain = df['Outcome'].values

print(np.shape(xtrain))
print(np.shape(ytrain))
print(xtrain)
print(ytrain)

model = tf.keras.Sequential(
    [
        keras.layers.Dense(12, activation='relu', input_shape=(8,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1,activation='sigmoid')
    ]
)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(xtrain,ytrain,epochs=200, verbose=1, batch_size=20)

