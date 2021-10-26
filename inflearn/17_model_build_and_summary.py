import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# # model Sequential
#
# model = Sequential()
# # model = add(InputLayer(input_shape=(28,28,1)))
# # batch를 설정하지 않는것이 핵심!
# model.add(Flatten())
# model.add(Dense(units=10))
# model.add(Activation('relu'))
# model.add(Dense(units=2))
# model.add(Activation('softmax'))
#
# model.build(input_shape=(None, 28,28,1))
# #input이 어떤 shape을 가지고 있는지, batch 사이즈가 달라질 수 있으니까 none으로 설정
#
# print(model.summary())

# sub classing
class TestModel(Model) :
    def __init__(self):
        super(TestModel,self).__init__()

        self.flatten = Flatten()
        self.d1 = Dense(units=10)
        self.d1_act = Activation('relu')
        self.d2 = Dense(units=2)
        self.d2_act = Activation('softmax')

    def call(self,x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d1_act(x)
        x = self.d2(x)
        x = self.d2_act(x)

        return x

model = TestModel()
model.build(input_shape=(None,28,28,1))

model.summary()


# model build 확인
model = Sequential()
model.add(Flatten())
model.add(Dense(units=10))
model.add(Activation('relu'))
model.add(Dense(units=2))
model.add(Activation('softmax'))

test_img = tf.random.normal(shape=(1,28,28,1))
model(test_img)