import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler

model = tf.keras.Sequential(
    [

        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(600, activation='relu'),
        keras.layers.Dense(200, activation='relu'),


        keras.layers.Dense(1)

    ]
)

model.compile( loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

# xs = np.array([0b0010,0b0011,0b0100,0b0101,0b0110], dtype=float) #input = 2,3,4,5,6
#
# ys = np.array([0b0100,0b0110,0b1000,0b1010,0b1100], dtype=float) #output = 4,6,8,10,12

ys = np.array([0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
            0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A,
            0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F])
xs = np.array([0xbde36565, 0xbde06565, 0xbde16565, 0xbde66565, 0xbde76565,
              0xbde46565, 0xbde56565, 0xbdea6565, 0xbdeb6565, 0xbde86565, 0xbde96565,
              0xbdee6565, 0xbdef6565, 0xbdec6565, 0xbded6565, 0xbdf26565, 0xbdf36565, 0xbdf06565,
              0xbdf16565, 0xbdf66565, 0xbdf76565, 0xbdf46565, 0xbdf56565, 0xbdfa6565, 0xbdfb6565, 0xbdf86565,
              0xbdc36565, 0xbdc06565, 0xbdc16565, 0xbdc66565, 0xbdc76565, 0xbdc46565, 0xbdc56565, 0xbdca6565,
              0xbdcb6565, 0xbdc86565, 0xbdc96565, 0xbdce6565, 0xbdcf6565, 0xbdcc6565, 0xbdcd6565])

print(np.shape(xs))
print(np.shape(ys))
print(xs, xs.dtype)
print(ys, xs.dtype)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

fit1 = model.fit(xs,ys, epochs=1500)

result = model.predict(np.array([0b1111],dtype=float)) #predic = 7
print(result)
result = np.round(result)

print(result)


for i in result:
    for a in i:
        print(a)
    a = np.array(a,dtype=int)

print(a.dtype , a, a.shape)
print(np.binary_repr(a))

