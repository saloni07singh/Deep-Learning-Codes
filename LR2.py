import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([500,700,900,1100,1300,1500,1700],dtype=float)
Y = np.array([150,200,250,300,350,400,450],dtype=float)

X = X/1700
Y = Y/450

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1,input_shape=[1])
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='mse'
)

history = model.fit(X,Y,epochs=200,verbose=0)

plt.plot(history.history['loss'])
plt.title("Loss")
plt.show()

pred=model.predict(X)

plt.scatter(X,Y)
plt.plot(X,pred,color='red')
plt.show()

new=np.array([1200/1700])

print(model.predict(new)*450)