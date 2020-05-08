import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Activation, Reshape
from tensorflow.keras import Model, backend
import numpy as np
import pickle
import sys

simplify_clusters_shape = pickle.load(open('x/x_norm.p', "rb"))
person_clusters_shape = pickle.load(open('y/y_norm.p', "rb"))

x_train = simplify_clusters_shape[:-100]
y_train = person_clusters_shape[:-100]

x_test = simplify_clusters_shape[-100:]
y_test = person_clusters_shape[-100:]

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

print(x_train.shape)
print(y_train.shape)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.relu = Activation('relu')
        self.reshape = Reshape((5, 20, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(40)
        self.dense2 = Dense(200)

    def decode(self, x):
        tf.print("input.shape", x.shape, output_stream=sys.stdout)
        tf.print("input", x, output_stream=sys.stdout)
        x = self.flatten(x)
        tf.print("flatten.shape", x.shape, output_stream=sys.stdout)
        tf.print("flatten", x, output_stream=sys.stdout)
        x = self.dense1(x)
        tf.print("dense1.shape", x.shape, output_stream=sys.stdout)
        tf.print("dense1", x, output_stream=sys.stdout)
        x = self.relu(x)
        tf.print("relu", x, output_stream=sys.stdout)
        x = self.dense2(x)
        tf.print("dense2", x, output_stream=sys.stdout)
        x = self.relu(x)
        tf.print("relu", x, output_stream=sys.stdout)
        x = self.reshape(x)
        tf.print("reshape", x, output_stream=sys.stdout)
        return x

        # x = self.flatten(x)
        # x = self.dense1(x)
        # x = self.relu(x)
        # x = self.dense2(x)
        # x = self.relu(x)
        # x = self.reshape(x)
        # return x

    def call(self, x):
        return self.decode(x)


optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def train_step(model, images, labels, loss_object):
    with tf.GradientTape() as tape:
        predictions = model(images)
        # print("predictions.shape", predictions.shape)
        # tf.print("predictions", predictions, output_stream=sys.stdout)
        # print()
        # print("image.shape", images.shape)
        # tf.print("images", images, output_stream=sys.stdout)
        # print()
        loss = loss_object(tf.cast(labels, tf.float32), tf.cast(predictions, tf.float32))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)


def train_model(model, epochs, loss_object):

    for epoch in range(epochs):
        for images, labels in train_ds:
            train_step(model, images, labels, loss_object)

        print(f'Epoch {epoch + 1}, Loss: {train_loss.result()}\n -- ')
        train_loss.reset_states()


def visualize_reconstruction(images, model):
    print(images[0].squeeze())
    reconstruction = model(images[0][tf.newaxis, :, :]).numpy().squeeze()
    print(reconstruction)


def loss_object_func(labels, predictions):
    # tf.print("predictions", predictions[:1,:1,:3,:], output_stream=sys.stdout)
    # tf.print("labels", labels[:1,:1,:3,:], output_stream=sys.stdout)
    # tf.print("square", tf.square(labels[:1, :1, :3, :] - predictions[:1, :1, :3, :]), output_stream=sys.stdout)
    # tf.print("sum+square", tf.keras.backend.sum(tf.square(labels[:1, :1, :3, :] - predictions[:1, :1, :3, :]), axis=3),
    #          output_stream=sys.stdout)
    # tf.print("sum+square+sqrt",
    #          tf.sqrt(tf.keras.backend.sum(tf.square(labels[:1, :1, :3, :] - predictions[:1, :1, :3, :]), axis=3)),
    #          output_stream=sys.stdout)
    # tf.print("loss", tf.keras.backend.sum(tf.sqrt(tf.keras.backend.sum(tf.square(labels - predictions), axis=3))),
    #          output_stream=sys.stdout)
    return tf.keras.backend.sum(tf.sqrt(tf.keras.backend.sum(tf.square(labels - predictions), axis=3)))


def run():
    model = MyModel()
    train_model(model, 500, loss_object_func)
    i = np.random.randint(0, len(x_test), size=5)
    images = x_test[i]
    visualize_reconstruction(images, model)

run()