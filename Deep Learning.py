import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler




model = tf.keras.Sequential(
    [
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(512, activation='relu'),

        keras.layers.Dense(1)
    ]
)



class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 0.01):
            print("\n\n\nReached 0.05 loss value so cancelling training!\n\n\n")

            self.model.stop_training = True




stop = haltCallback()

#model.load_weights('key3/')



model.compile( loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

xs = np.array([[32,92],[33,33],[34,13],[35,106],[36,25],[37,68],[38,104],[39,207],[40,114],[41,111],
               [42,119],[43,248],[44,247],[45,202],[46,242],[47,189],[48,18],[49,66],[50,156],[51,123],
               [52,151],[53,199],[54,217],[55,58],[56,121],[57,140],[58,55],[59,103],[60,60],[61,233],
               [62,178],[63,226],[64,64],[65,29],[66,49],[67,214],[68,37],[69,88],[70,112],[71,83],[72,206],
               [73,243],[74,235],[75,228],[76,107],[77,118],[78,78],[79,129],[80,174],[81,186],[82,128],[83,71],
               [84,11],[85,251],[86,229],[87,194],[88,69],[89,148],[90,171],[91,91],[92,32],[93,213],[94,14],
               [95,26],[96,232],[97,181],[98,153],[99,122],[100,141],[101,244],[102,220],[103,59],[104,38],
               [105,155],[106,35],[107,76],[108,163],[109,218],[110,166],[111,41],[112,70],[113,22],[114,40],
               [115,143],[116,195],[117,147],[118,77],[119,42],[120,237],[121,56],[122,99],[123,51],[124,136],
               [125,125],[126,230]
],dtype=int)

ys = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,], dtype=int)

print(np.shape(xs))
print(np.shape(ys))
print(xs, xs.dtype)
print(ys, ys.dtype)


monitor = EarlyStopping(monitor='loss',patience=20000, verbose=1 ,mode='min' , restore_best_weights=True)

fit1 = model.fit(xs,ys, epochs=200000, callbacks=[stop])

#model.save_weights('key3/')


result = model.predict(np.array([[106,54],[107,166],[108,138],[109,109],[110,83],[111,227],[112,140],[113,3],
               [114,155],[115,198],[116,9],[117,70],[118,254],[119,131],[120,94],[121,121],[122,21],[123,40],
               [124,59],[125,252],[126,176]],dtype=int))

#result = np.round(result)

print(result)



