import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, Activation, Reshape, LeakyReLU
from tensorflow.keras import Model
import numpy as np
import pickle
import matplotlib.pyplot as plt

simplify_clusters_shape = pickle.load(open('x/x.p', "rb"))
person_clusters_shape = pickle.load(open('y/y.p', "rb"))

x_train = simplify_clusters_shape[:-100]
y_train = person_clusters_shape[:-100]

x_test = simplify_clusters_shape[-100:]
y_test = person_clusters_shape[-100:]

# x_train = x_train[..., tf.newaxis]
# x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

print(x_train.shape)
print(y_train.shape)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # OPS
        self.relu = Activation('relu')
        self.sigmoid = Activation('sigmoid')
        self.reshape = Reshape((5, 20, 2))

        # Fully connected layers
        self.flatten = Flatten()
        self.dense1 = Dense(200)

    def decode(self, x):
        # activate the decoder part of the network
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.reshape(x)
        x = self.sigmoid(x)
        return x

    def call(self, x):
        return self.decode(x)


optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def train_step(model, images, labels, loss_object):

    with tf.Session() as sess:  print(labels[0].eval())
    print()
    print(images[0].eval())
    exit(1)

    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    # train_accuracy(labels, predictions)


def train_model(model, epochs, loss_object):

    for epoch in range(epochs):
        for images, labels in train_ds:
            train_step(model, images, labels, loss_object)

            template = 'Epoch {}, Loss: {}, Accuracy: {}'
        # print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100))
        # print(" -- ", flush=True)

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        # train_accuracy.reset_states()

def visualize_reconstruction(images, model):
    print(images[0].squeeze())
    reconstruction = model(images[0][tf.newaxis, :, :]).numpy().squeeze()
    print(reconstruction)
    # i = 1
    # j = 1
    # plt.figure(figsize=(10, 10))
    # for im in images:
    #     reconstruction = model(im[tf.newaxis, :, :, :]).numpy().squeeze()
    #     print(reconstruction)

        # plt.subplot(len(images), 2, i)
        # plt.imshow(im.squeeze(), cmap=plt.get_cmap('gray'))
        # plt.title(f'original image {j}')
        # plt.subplot(len(images), 2, i + 1)
        # plt.imshow(reconstruction, cmap=plt.get_cmap('gray'))
        # plt.title(f'image {j} reconstruction')
        # i += 2
        # j += 1
    # plt.subplots_adjust(hspace=0.8)
    # plt.show()


def run():
    loss_object = tf.keras.losses.binary_crossentropy
    model = MyModel()
    train_model(model, 5, loss_object)
    i = np.random.randint(0, len(x_test), size=5)
    images = x_test[i]
    visualize_reconstruction(images, model)

run()