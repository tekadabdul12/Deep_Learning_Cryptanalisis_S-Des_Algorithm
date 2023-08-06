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


        keras.layers.Dense(10)

    ]
)

model.compile( loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

# xs = np.array([0b0010,0b0011,0b0100,0b0101,0b0110], dtype=float) #input = 2,3,4,5,6
#
# ys = np.array([0b0100,0b0110,0b1000,0b1010,0b1100], dtype=float) #output = 4,6,8,10,12

xs = np.array([[0x1410, 0x41], [0x1420, 0x42], [0x1430, 0x43], [0x1440, 0x44],
               [0x1450, 0x45], [0x1460, 0x46], [0x1470, 0x47], [0x1480, 0x48],
               [0x1490, 0x49], [0x14a0, 0x4a], [0x14b0, 0x4b], [0x14c0, 0x4c],
               [0x14d0, 0x4d], [0x14e0, 0x4e], [0x14f0, 0x4f], [0x1500, 0x50],
               [0x1510, 0x51], [0x1520, 0x52], [0x1530, 0x53], [0x1540, 0x54],
               [0x1550, 0x55], [0x1560, 0x56], [0x1570, 0x57], [0x1580, 0x58],
               [0x1590, 0x59], [0x15a0, 0x5a]])

ys = np.array([0x0100,0x0100,0x0100,0x0100,
               0x0100,0x0100,0x0100,0x0100,
               0x0100,0x0100,0x0100,0x0100,
               0x0100,0x0100,0x0100,0x0100,
               0x0100,0x0100,0x0100,0x0100,
               0x0100,0x0100,0x0100,0x0100,
               0x0100,0x0100,

])

class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 0.2):
            print("\n\n\nReached 0.05 loss value so cancelling training!\n\n\n")

            self.model.stop_training = True

stop = haltCallback()


print(np.shape(xs))
print(np.shape(ys))
print(xs, xs.dtype)
print(ys, xs.dtype)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=200)

fit1 = model.fit(xs,ys, epochs=7000, callbacks=[stop])

#result = model.predict(np.array([0b1111],dtype=float)) #predic = 7
# print(result)
# result = np.round(result)

# print(result)
#
#
# for i in result:
#     for a in i:
#         print(a)
#     a = np.array(a,dtype=int)
#
# print(a.dtype , a, a.shape)
# print(np.binary_repr(a))

