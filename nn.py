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

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)

print(x_train.shape)
print(y_train.shape)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.reshape = Reshape((5, 20, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(200, activation='relu')
        self.dense2 = Dense(200, activation='relu')
        self.dense3 = Dense(1000, activation='relu')
        self.dense4 = Dense(200, activation='relu')
        self.dense5 = Dense(200)

    def decode(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.reshape(x)
        return x

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


def visualize_reconstruction(simplified_test, label_test, model):
    import matplotlib.pyplot as plt

    sim = np.array(simplified_test)
    plt.subplot(131)
    plt.title("simplify")
    plt.plot(sim[:,0], sim[:,1], 'o')

    lab = np.array(label_test)
    plt.subplot(132)
    plt.title("label")
    for i in range(5):
        plt.plot(lab[i][:, 0], lab[i][:, 1])

    reconstruction = model(simplified_test[tf.newaxis, :, :]).numpy().squeeze()
    rec = np.array(reconstruction)
    plt.subplot(133)
    plt.title("style")
    for j in range(5):
        plt.plot(rec[j][:, 0], rec[j][:, 1])

    plt.show()

def loss_object_func(labels, predictions):
    # tf.print("predictions", predictions[:1,:1,:3,:], output_stream=sys.stdout)
    # tf.print("labels", labels[:1,:1,:3,:], output_stream=sys.stdout)
    # tf.print("square", tf.square(labels[:1, :1, :3, :] - predictions[:1, :1, :3, :]), output_stream=sys.stdout)
    # tf.print("sum+square", tf.keras.backend.sum(tf.square(labels[:1, :1, :3, :] - predictions[:1, :1, :3, :]), axis=3),
    #          output_stream=sys.stdout)
    # tf.print("sum+square+sqrt",
    #          tf.sqrt(tf.keras.backend.sum(tf.square(labels[:1, :1, :3, :] - predictions[:1, :1, :3, :]), axis=3)),
    #          output_stream=sys.stdout)
    # tf.print("loss", tf.keras.backend.sum(tf.sqrt(tf.keras.backend.sum(tf.square(labels[:1, :1, :3, :] - predictions[:1, :1, :3, :]), axis=3))),
    #          output_stream=sys.stdout)
    return tf.keras.backend.sum(tf.sqrt(tf.keras.backend.sum(tf.square(labels - predictions), axis=3)))


def run():
    model = MyModel()
    train_model(model, 10, loss_object_func)
    model.summary()
    for j in range(3):
        i = np.random.randint(0, len(x_test))
        simplified_test = x_test[i]
        label_test = y_test[i]
        visualize_reconstruction(simplified_test, label_test, model)

# run()