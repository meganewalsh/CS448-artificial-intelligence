from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(class_names)

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential()
# input: 1 image at 28x28
input_shape = (28, 28, 1)
# first conv filter: 3x1x5x5 3@24x24
model.add(keras.layers.Conv2D(3, kernel_size = (5, 5), strides=(1, 1), activation= 'relu', input_shape=input_shape, padding = 'valid'))
# first max pooling layer 2x, 3@12x12
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2)))
#second conv filter: 3x3x3x3 3@12@12
model.add(keras.layers.Conv2D(3, kernel_size = (3, 3), strides=(1, 1), activation = 'relu', padding = 'same'))
#second max pooling layer 2x, 3@6x6
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides = (2)))
# flatten to 108@1x1
model.add(keras.layers.Flatten())
# dense layer 100@1x1
model.add(keras.layers.Dense(100, activation = 'relu'))
# dense layer 50@1x1
model.add(keras.layers.Dense(50, activation = 'relu'))
# dense layer 10@1x1
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_images = train_images[:, :, :, np.newaxis]
test_images = test_images[:, :, :, np.newaxis]

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)

model.summary()
print('Test accuracy:', test_acc)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model.save('keras_model.h5')
new_model = keras.models.load_model('model.h5')
loss, acc = new_model.evaluate(test_images, test_labels)
