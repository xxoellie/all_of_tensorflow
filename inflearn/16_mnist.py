import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Activation

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
from termcolor import colored


def get_mnist_ds():
    (train_validation_ds, test_ds) , ds_info = tfds.load(name='mnist',
                                                        shuffle_files=True,
                                                        as_supervised=True,
                                                        split=['train','test'],
                                                       with_info=True)
    # 전체 데이터 개수를 가져와야 하므로 ds_info 가져와야함
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
    validation_ds = validation_ds.map(stnd).batch(TEST_BATCH_SIZE)
    test_ds = test_ds.map(stnd).batch(TEST_BATCH_SIZE)

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

    train_acc = SparseCategoricalAccuracy()
    validation_acc = SparseCategoricalAccuracy()
    test_acc = SparseCategoricalAccuracy()

@tf.function
def trainer():
    global train_ds, model, loss_object, optimizer
    global train_loss, train_acc

    for images, labels in train_ds :
        with tf.GradientTape() as tape :
            predictions = model(images)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))

        train_loss(loss)
        train_acc(labels, predictions)

@tf.function
def validation():
    global validation_ds, model, loss_object
    global validation_loss, validation_acc

    for images, labels in train_ds:
        predictions = model(images)
        loss = loss_object(labels, predictions)


        validation_loss(loss)
        validation_acc(labels, predictions)

@tf.function
def tester():

    global test_ds, model, loss_object
    global test_loss, test_acc

    for images, labels in train_ds:
        predictions = model(images)
        loss = loss_object(labels, predictions)

        test_loss(loss)
        test_acc(labels, predictions)

def train_reporter():
    global epoch
    global train_loss, train_acc
    global validation_loss, validation_acc
    global test_loss, test_acc

    print(colored('Epoch', 'red', 'on_white'), epoch+1)
    template = "Train loss : {:.4f}\t Train Accuaracy : {:.2f}%\n" +\
        "validation loss : {:.4f}\t validation Accuaracy : {:.2f}%\n"

    print(template.format(train_loss.result(), train_acc.result()*100,
                          validation_loss.result(), validation_acc.result()*100))

    train_acc.reset_states()
    train_loss.reset_states()
    validation_loss.reset_states()
    validation_acc.reset_states()

EPOCHS = 10
LR = 0.001
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 32

train_ds, validation_ds, test_ds = get_mnist_ds()
standardization(TRAIN_BATCH_SIZE,TEST_BATCH_SIZE)

model = MNIST_Classifier()
loss_object = SparseCategoricalCrossentropy()
optimizer = SGD(learning_rate=LR)

load_metrics()

for epoch in range(EPOCHS):
    trainer()
    validation()
    train_reporter()

tester()