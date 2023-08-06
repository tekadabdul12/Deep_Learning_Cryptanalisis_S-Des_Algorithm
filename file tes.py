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
        keras.layers.Dense(800, activation='relu'),
        keras.layers.Dense(800, activation='relu'),
        keras.layers.Dense(800, activation='relu'),
        keras.layers.Dense(800, activation='relu'),
        keras.layers.Dense(800, activation='relu'),
        keras.layers.Dense(800, activation='relu'),
        keras.layers.Dense(800, activation='relu'),
        keras.layers.Dense(800, activation='relu'),
        keras.layers.Dense(800, activation='relu'),
        keras.layers.Dense(800, activation='relu'),
        keras.layers.Dense(800, activation='relu'),
        keras.layers.Dense(800, activation='relu'),
        keras.layers.Dense(800, activation='relu'),
        keras.layers.Dense(800, activation='relu'),
        keras.layers.Dense(800, activation='relu'),






        keras.layers.Dense(1)
    ]
)

path = "model1.ckpt"


class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 2):
            print("\n\n\nReached 0.05 loss value so cancelling training!\n\n\n")
            tf.keras.callbacks.ModelCheckpoint(
                filepath = path,
                monitor='loss',
                verbose=1,
                save_weights_only=True
            )
            self.model.stop_training = True




stop = haltCallback()

model.load_weights('model4/')

model.compile( loss='mean_squared_error', optimizer='RMSprop')

#xs = np.array([0b00011001,0b10100000,0b00100110,0b11000101,0b00100101,0b10010100,0b01001000,0b11101010], dtype=int)
#ys = np.array([0b00001000,0b11011110,0b11000000,0b11101111,0b10111110,0b00000001,0b10000000,0b00010000], dtype=int)


xs = np.array([[91,341],[248,341],[141,341],[55,341],[78,341],[185,341],[40,341],[118,341],[11,341],[126,341],[3,341],
               [178,341],[10,341],[232,341],[2,341],[243,341],[79,341],[77,341],[233,341],[221,341],[112,341],[83,341],
               [76,341],[156,341],[245,341],[18,341],[196,341],[17,341],[15,341],[114,341],[97,341],[148,341],[74,341],
               [51,341],[98,341],[63,341],[193,341]], dtype=int) #input = a-z kecuali a,m,z
ys = np.array([32,48,49,50,51,52,53,54,55,56,57,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,
               89,90], dtype=int) #output = a-z

print(np.shape(xs))
print(np.shape(ys))
print(xs, xs.dtype)

print(ys, xs.dtype)


monitor = EarlyStopping(monitor='loss',patience=20000, verbose=1 ,mode='min' , restore_best_weights=True)

fit1 = model.fit(xs,ys, epochs=100000, callbacks=[stop])

#model.save_weights('model4/')

#result = model.predict([0b11000110])
result = model.predict(np.array([[5,20]],dtype=int)) #predic = 7

result = np.round(result)

print(result)


# for i in result:
#     for a in i:
#         print(a)
#     a = np.array(a,dtype=int)

# print(a.dtype , a, a.shape)
# print(np.binary_repr(a))

