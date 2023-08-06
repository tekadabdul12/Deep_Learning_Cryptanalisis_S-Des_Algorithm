import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler



'''
df =pd.read_csv('diabetes.csv')
#print(df.head())

xtrain = df.values
xtrain = np.delete(xtrain,8,axis=1)

#print(xtrain)

xtrain = MinMaxScaler().fit_transform(xtrain)


ytrain = df['Outcome'].values

print(np.shape(xtrain))
print(np.shape(ytrain))
print(ytrain)

model = tf.keras.Sequential(
    [
        keras.layers.Dense(12, activation='relu', input_shape=(8,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1,activation='sigmoid')
    ]
)

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.fit(xtrain,ytrain,epochs=200, verbose=1, batch_size=20)

'''




model = tf.keras.Sequential(
    [
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),


        keras.layers.Dense(1)
    ]
)

# path = "model1.ckpt"


class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 5):
            print("\n\n\nReached 0.05 loss value so cancelling training!\n\n\n")

            self.model.stop_training = True




stop = haltCallback()

#model.load_weights('model9_k000000/')



model.compile( loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

#xs = np.array([0b00011001,0b10100000,0b00100110,0b11000101,0b00100101,0b10010100,0b01001000,0b11101010], dtype=int)
#ys = np.array([0b00001000,0b11011110,0b11000000,0b11101111,0b10111110,0b00000001,0b10000000,0b00010000], dtype=int)


# xs = np.array([67,24,205,150,6,157,72,203,232,54,166,138,109,83,227,140,3,155,198,9,70,254,131,94,121,21,], dtype=int) #input = a-z kecuali a,m,z
#
# ys = np.array([97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122], dtype=int) #output = a-z

ys = np.array([97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,
               112,113,114,115,116,117,118,119,120,121,122], dtype=int)

xs = np.array([146,140,89,199,215,201,60,105,138,103,119,
               44,11,226,50,238,165,239,58,111,228,110,159,112,63,1], dtype=int)

print(np.shape(xs))
print(np.shape(ys))
print(xs, xs.dtype)

print(ys, xs.dtype)


monitor = EarlyStopping(monitor='loss',patience=100, verbose=1 ,mode='min' , restore_best_weights=True)

fit1 = model.fit(xs,ys, epochs=10000, callbacks=[stop])

#model.save_weights('k1001111000/')

#result = model.predict([0b11000110])
result = model.predict(np.array([67,24,205,150,6,157,72,203,232,54,166,138,109,83,227,140,3,155,198,9,70,254,131,94,121,21,],dtype=int)) #predic = 7

result = np.round(result)

print(result)


# for i in result:
#     for a in i:
#         print(a)
#     a = np.array(a,dtype=int)

# print(a.dtype , a, a.shape)
# print(np.binary_repr(a))

