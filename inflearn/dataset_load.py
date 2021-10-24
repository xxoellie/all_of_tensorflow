import numpy as np
import tensorflow as tf
import sys

from tensorflow.python.eager import context

train_x = np.arange(1000).astype(np.float32).reshape(-1,1)
train_y = 3*train_x + 1

train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_ds = train_ds.shuffle(100).batch(32)

# for x, y in train_ds :
#     print(x.shape, y.shape)

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import tensorflow_datasets as tfds
from tensorflow.data import Dataset
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print(train_images.shape)
# print(test_images.shape)
# print(train_labels.shape)
# print(test_labels.shape)
#
# print(sys.getsizeof(train_images))
# print(sys.getsizeof(train_images)/1024/1024)

train_ds = Dataset.from_tensor_slices((train_images,train_labels))
train_ds = train_ds.shuffle(60000).batch(9)


test_ds = Dataset.from_tensor_slices((test_images, test_labels))
test_ds = test_ds.batch(32)

# train_ds_iter = iter(train_ds)
# images, labels = next(train_ds_iter)

iterator = train_ds.make_one_shot_iterator()
images, labels = iterator.get_next()


print(images.shape)
print(labels.shape)

fig, axes = plt.subplots(3,3, figsize=(10,10))

for ax_idx, ax in enumerate(axes.flatten()) :
    image = images[ax_idx, ... ]
    label = labels[ax_idx]

    ax.imshow(image.numpy(), 'gray')
    ax.set_title(label.numpy(), 'gray')

    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    plt.show()