from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from model import MyModel

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
'''
tensorflow中图片类型为[B,H,W,C]
Pytorch中图片的类型为[B,C,H,W]
'''
# ============================================= 【download and load data】 ============================================ #
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 下载后的路径：C:\Users\BushiZ\.keras\datasets\mnist.npz
x_train, x_test = x_train / 255.0, x_test / 255.0

# ================================================== 【show images】 ================================================= #
'''
import numpy as np
import matplotlib.pyplot as plt

test_imgs = x_test[:3]
test_labs = y_test[:3]
print(test_labs)
plot_imgs = np.hstack(test_imgs)    
plt.imshow(plot_imgs, cmap='gray')
plt.show()
'''

# ============================================ 【Add a channels dimension】 =========================================== #
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# ============================================= 【create data generator】 ============================================= #
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# ============================================= 【define loss\optimizer\mode】 ============================================= #
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
model = MyModel()

# ====================================== 【define train_loss and train_accuracy】 ===================================== #
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')  # acc = accuracy
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# ====================================== 【define train_loss and train_accuracy】 ===================================== #
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    loss = loss_object(labels, predictions)

    test_loss(loss)
    test_acc(labels, predictions)

# ====================================== 【define train_loss and train_accuracy】 ===================================== #
Epoch = 5
for epoch in range(Epoch):
    train_loss.reset_states()     # clear history info
    test_loss.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()

    for train_images, train_labels in train_ds:
        train_step(train_images, train_labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print = 'Epoch : {}, loss : {}, Accuracy : {}, Test_loss : {}, Test_Accuracy : {} '
    print(print.format(epoch + 1,
                       train_loss.result(),
                       train_acc.result() * 100,
                       test_loss.result(),
                       test_acc.result() * 100))