import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Activation

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.metrics import Mean, SparseCategoricalCrossentropy

def get_mnist_ds():
    (train_validation_ds, test_ds) , ds_info = tfds.load(name='mnist',
                                                        shuffle_files=True,
                                                        split=['train','test'],
                                                       with_info=True)
    n_train_validation = ds_info.splits['train'].num_examples

    train_ratio = 0.8
    n_train = int(n_train_validation * train_ratio)
    n_validation = n_train_validation - n_train

    train_ds = train_validation_ds.take(n_train)
    remaining_ds = train_validation_ds.skip(n_train)
    validation_ds = remaining_ds.take(n_validation)

    return train_ds, validation_ds, test_ds

train_ds, validation_ds, test_ds = get_mnist_ds()

# print(train_ds)
# print(validation_ds)
# print(test_ds)

# standardization

def standardization(TRAIN_BATCH_SIZE, TEST_BATCH_SIZE) :
    global train_ds, validation_ds, test_ds

    def stnd(images, labels):
        images = tf.cast(images, tf.float32) / 255.
        return [images, labels]

    train_ds = train_ds.map(stnd).shuffle(1000).batch(TRAIN_BATCH_SIZE)
    validation_ds = validation_ds.map(stnd).batch(TRAIN_BATCH_SIZE)
    test_ds = test_ds.map(stnd).batch(TRAIN_BATCH_SIZE)

class MNIST_Classifier(Model):
    def __init__(self):
        super(MNIST_Classifier, self).__init__()

        self.flatten = Flatten()
        self.d1 = Dense(64, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self,x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)

        return x

    def load_metrics():
        global train_loss, train_acc
        global validation_loss, validation_acc
        global test_loss, test_acc

        train_loss = Mean()
        validation_loss = Mean()
        test_loss = Mean()

        train_acc, SparseCategoricalCrossentropy()