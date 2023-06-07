import numpy as np
import tensorflow as tf
import keras as k

train_x = np.loadtxt()

model = k.models.Sequential()
model.add(k.layers.Dense(units=7,input_dim=4,activation='tanh'))

model.add(k.layers.Dense(units=3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

print('Starting training')
h = model.fit(train_x, train_y,baych_size=1, epochs=12, verbose=1)
print('Training finished')

eval = model.evaluate(train_x,train_y, verbose=0)
print('evaluation Loss = ', eval[0], eval[1]*100)

np.set_printoptions(precision=4)
u = np.array([[6.1, 3.1, 5.1, 1.1]], dtype=np.float32)

