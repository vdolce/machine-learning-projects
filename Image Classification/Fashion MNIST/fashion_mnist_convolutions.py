# -*- coding: utf-8 -*-
"""
Fashion MNIST
dataset in tf.keras.datasets.mnist

@author: Valeria Dolce
"""
import tensorflow as tf
from os import path, getcwd, chdir

DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
path = tf.keras.utils.get_file('mnist.npz', DATA_URL)

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():

    class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') >= 0.998):
          print("\nReached 99.8% accuracy so cancelling training!")
          self.model.stop_training = True

    


    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    callbacks = myCallback()

    # reshape images
    training_images=training_images.reshape(60000, 28, 28, 1)
    training_images=training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images=test_images/255.0

    # the model
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    # model fitting
    history = model.fit(
              test_images, 
              test_labels, 
              epochs=10, 
              callbacks=[callbacks]
    )

    # model fitting
    return history.epoch, history.history['acc'][-1]

_, _ = train_mnist_conv()