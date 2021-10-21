import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# tf1 = tf.Variable([1,2,3], dtype=tf.float32)
# tf2 = tf.Variable([10,20,30], dtype=tf.float32)
#
# with tf.GradientTape() as tape :
#     tf3 = tf1 * tf2
#     tf4 = tf3 + tf2
#
# gradients = tape.gradient(tf4, [tf1,tf2,tf3])
# print(gradients[0])
# print(gradients[1])
# print(gradients[2])


x_data = tf.random.normal(shape=(1000,), dtype=tf.float32)
y_data = 3*x_data + 1

w = tf.Variable(-1.)
b = tf.Variable(-1.)

LR=0.01
EPOCHS=10
w_trace, b_trace = [], []
for epoch in range(EPOCHS) :
    for x, y in zip(x_data, y_data) :
        with tf.GradientTape() as tape :
            prediction = w*x + b
            loss = (y - prediction) ** 2

        gradients = tape.gradient(loss, [w,b])

        w_trace.append(w.numpy())
        b_trace.append(b.numpy())
        w = tf.Variable(w - LR*gradients[0])
        b = tf.Variable(b - LR*gradients[1])

fig, ax = plt.subplots(figsize=(20,10))

ax.plot(w_trace,
        label='weight')
ax.plot(b_trace,
        label='bias')
ax.tick_params(labelsize=20)

ax.legend(fontsize=30)

ax.grid()

plt.show()


# print(tf.__version__)
#
# t1 = tf.Variable([1,2,3])
# t2 = tf.constant([1,2,3])
#
# print(t1)
# print(t2)

# test_list = [1,2,3]
# test_np = np.array([1,2,3])
#
# t1 = tf.constant(test_list)
# t2 = tf.constant(test_np)
#
# print(t1)
# print(t2)
#
# t3 = tf.Variable(test_list)
# t4 = tf.Variable(test_np)
#
# print(t3)
# print(t4)


