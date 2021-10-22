import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Mean


import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored

n_train = 1000

train_x = np.random.normal(0,1,size=(n_train,1)).astype(np.float32)
train_x_noise = train_x + 0.2 * np.random.normal(0,1,size=(n_train,1))

train_y = (train_x_noise> 0).astype(np.int32)

# fig, ax = plt.subplots(figsize=(15,10))
# ax.scatter(train_x, train_y)
# ax.tick_params(labelsize=20)
# ax.grid()
# plt.show()



train_ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))
train_ds = train_ds.shuffle(n_train).batch(8)

model = Sequential()
model.add(Dense(units=2, activation='softmax'))


# class Mymodel(Model):
#     def __init__(self):
#         super(Mymodel, self).__init__()
#         self.d1 = Dense(units=2, activation='softmax')
#
#     def call(self,x):
#         x= self.d1(x)
#         return x

loss_object = SparseCategoricalCrossentropy()
optimizer = SGD(learning_rate=1)

train_loss = Mean()
train_acc = SparseCategoricalAccuracy()

EPOCHS = 10

for epoch in range(EPOCHS) :
    for x,y in train_ds :
        with tf.GradientTape() as tape :
            predictions = model(x)
            loss = loss_object(y, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # 순전파 방식 적용

        train_loss(loss)
        train_acc(y, predictions)

    print(colored('Epoch :', 'red','on_white'), epoch+1)
    template = 'Train loss : {:.4f}\t Train Accuracy : {:.2f}%\n'
    print(template.format(train_loss.result(),
                          train_acc.result() * 100))

    train_loss.reset_states()
    train_acc.reset_states()


#%%

train_loss = Mean()
t1 = tf.constant([1,2,3,4,5,6])
for t in t1 :
    train_loss(t)
    print(train_loss.result())

train_loss.reset_states()
#reset해주므로서 동일하게 작동

t2= tf.constant([1,2,3,4,5,6])
for t in t2 :
    train_loss(t)
    print(train_loss.result())

# 누적됬던 loss값이 살아남