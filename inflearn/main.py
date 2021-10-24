import tensorflow as tf
from typing import List, Tuple
import tensorflow_datasets as tfds

# https://www.tensorflow.org/datasets/catalog/overview


def build_model(input_shape):
    input_layer = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3))(input_layer)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3))(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(units=10)(x)
    x = tf.keras.layers.Softmax()(x)
    return tf.keras.Model(inputs=input_layer, outputs=x)



# https://www.google.com/search?q=MNIST&oq=MNIST&aqs=chrome..69i57j35i39j69i60l5j69i65.784j0j7&sourceid=chrome&ie=UTF-8

def preprocess(data):
    x, y = data["image"], data["label"]
    x = tf.cast(x, dtype=tf.float32)
    y = tf.one_hot(y, 10)
    return x, y


def get_data():
    data = tfds.load("mnist")
    train_data, test_data = data["train"], data["test"]

    train_data = train_data.map(preprocess).batch(32)
    test_data = test_data.map(preprocess).batch(32)
    return train_data, test_data

def option1():
    input_shape = [28, 28, 1]

    train_data, val_data = get_data()
    model = build_model(input_shape=input_shape)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                  loss=tf.keras.losses.CategoricalCrossentropy())
    model.fit(train_data, validation_data=val_data)


def option2():
    input_shape = [28, 28, 1]

    train_dataset, val_dataset = get_data()
    model = build_model(input_shape=input_shape)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    for i, train_data in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            xs, ys = train_data
            output = model(xs, training=True)
            loss = loss_fn(y_true=ys, y_pred=output)
            if i % 50 == 0:
                print(f"Iteration {i}: Loss {loss:3.5f}")
            grads = tape.gradient(target=loss, sources=model.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))



def main():
    option2()


if __name__ == '__main__':
    main()
