import tensorflow as tf
import tensorflow_datasets as tfds
from keras.preprocessing.image import ImageDataGenerator

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

def preprocess(data):
    x, y = data["image"], data["label"]
    x = tf.cast(x, dtype=tf.float32) / 255
    x1 = (x - 0.5)*2
    y = tf.one_hot(y, 10)
    return x1, y

def get_data():
    data = tfds.load("mnist")
    train_data, test_data = data["train"], data["test"]

    train_data = train_data.map(preprocess).batch(128)
    test_data = test_data.map(preprocess).batch(128)
    return train_data, test_data

def data_augmentation(x1,y):
    x1 = tf.image.random_crop(x1)
    x1 = tf.image.random_brightness(x1, max_delta=0.5)
    return x1,y


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
