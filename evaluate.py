
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from tensorflow import keras

import pandas as pd

# variables and class map for converting pred to char
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
class_map = pd.read_csv('data/kmnist_classmap.csv')

# load test data
x_test = np.load('data/kmnist-test-imgs.npz')['arr_0']
y_test = np.load('data/kmnist-test-labels.npz')['arr_0']

classifications = 10

# format data
x_test = x_test.astype('float32')
x_test /= 255
x_test = x_test.reshape(x_test.shape[0], *input_shape)
y_test = to_categorical(y_test, classifications)

# load model
model = keras.models.load_model('kmnist_model.h5')

# get acc
loss, acc = model.evaluate(x_test, y_test)
print('acc: {}'.format(100 * acc))

# predict
predictions = model.predict(x_test[:1000], verbose=1)
prediction_test = predictions[2]

# map predicted number index to classmaps char
number = np.argmax(prediction_test)
print(class_map.loc[number, 'char'])

# show actual image
actual = x_test[2]
pixels = actual.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

