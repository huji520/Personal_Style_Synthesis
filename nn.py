import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Add, Activation, LeakyReLU
from tensorflow.keras import Model
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

simplify_clusters_shape = pickle.load(open('x/x40_10_simplify_1_40.p', "rb"))
person_clusters_shape = pickle.load(open('y/y40_10_simplify_1_40.p', "rb"))

# for i in range(0, 20):
#     plt.figure(i)
#     plt.subplot(121)
#     plt.title("simplify")
#     plt.xlim(-100, 100)
#     plt.ylim(-100, 100)
#     plt.plot(simplify_clusters_shape[i][:, 0], simplify_clusters_shape[i][:, 1], 'o')
#
#     lab = np.array(person_clusters_shape[i])
#     plt.subplot(122)
#     plt.title("cluster")
#     plt.xlim(-100, 100)
#     plt.ylim(-100, 100)
#     for j in range(5):
#         plt.plot(lab[j][:, 0], lab[j][:, 1])
#     plt.savefig(f'results/input/40_10_simplify_1_40/{i}.png')


indexes1 = np.arange(len(simplify_clusters_shape))
np.random.shuffle(indexes1)

simplify_clusters_shape = simplify_clusters_shape[indexes1]
person_clusters_shape = person_clusters_shape[indexes1]

x_train = simplify_clusters_shape[:-200]
y_train = person_clusters_shape[:-200]

x_test = simplify_clusters_shape[-200:]
y_test = person_clusters_shape[-200:]

x_train = x_train[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(16)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(16)

print(x_train.shape)
print(y_train.shape)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.reshape = Reshape((5, 20, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(40, activation='relu')
        self.dense2 = Dense(80, activation='relu')
        self.dense2 = Dense(160, activation='relu')
        self.dense3 = Dense(320, activation='relu')
        self.dense4 = Dense(320, activation='relu')
        self.dense5 = Dense(640, activation='relu')
        self.dense6 = Dense(320, activation='relu')
        self.dense7 = Dense(320, activation='relu')
        self.dense8 = Dense(200)

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        x = self.dense8(x)
        x = self.reshape(x)
        return x


optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')


@tf.function
def train_step(model, images, labels, loss_object):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(tf.cast(labels, tf.float32), tf.cast(predictions, tf.float32))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)


def test_step(model, images, labels, loss_object):
    predictions = model(images)
    loss = loss_object(tf.cast(labels, tf.float32), tf.cast(predictions, tf.float32))
    test_loss(loss)


def train_model(model, epochs, loss_object, graph_train=None, graph_test=None, save=True):

    for epoch in range(epochs):
        for images, labels in train_ds:
            train_step(model, images, labels, loss_object)

        for images_test, labels_test in test_ds:
            test_step(model, images_test, labels_test, loss_object)
        print(f'Epoch {epoch + 1}, Train loss: {train_loss.result()}, Test loss: {test_loss.result()}\n -- ')
        if save:
            if (epoch+1) % 10 == 0:
                model.save_weights(f"results/weights/{epoch+1}_reg.tf")
                for k in range(30):
                    simplified_test = x_test[k]
                    label_test = y_test[k]
                    visualize_reconstruction(simplified_test, label_test, model, k, epoch)

        if graph_train:
            graph_train.append(train_loss.result())
        if graph_test:
            graph_test.append(test_loss.result())
        train_loss.reset_states()
        test_loss.reset_states()


def visualize_reconstruction(simplified_test, label_test, model, k, epoch):
    sim = np.array(simplified_test)
    plt.figure(k)
    plt.subplot(221)
    plt.title("simplify")
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    plt.plot(sim[:,0], sim[:,1])

    plt.subplot(222)
    plt.title("simplify (points)")
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    plt.plot(sim[:, 0], sim[:, 1], 'o', ms=2)

    lab = np.array(label_test)
    plt.subplot(223)
    plt.title("label")
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    for i in range(5):
        plt.plot(lab[i][:, 0], lab[i][:, 1])

    reconstruction = model(simplified_test[tf.newaxis, :, :]).numpy().squeeze()
    rec = np.array(reconstruction)
    plt.subplot(224)
    plt.title("style")
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    for j in range(5):
        plt.plot(rec[j][:, 0], rec[j][:, 1])

    if not os.path.isdir(f'results/{epoch}'):
        os.mkdir(f'results/{epoch}')
    plt.savefig(f'results/{epoch}/{k}.png')


def visualize_train(simplified_train, label_train, model, k, epoch):
    sim = np.array(simplified_train)
    plt.figure(k)
    plt.subplot(221)
    plt.title("simplify")
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    plt.plot(sim[:,0], sim[:,1])

    plt.subplot(222)
    plt.title("simplify (points)")
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    plt.plot(sim[:, 0], sim[:, 1], 'o', ms=2)

    lab = np.array(label_train)
    plt.subplot(223)
    plt.title("label")
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    for i in range(5):
        plt.plot(lab[i][:, 0], lab[i][:, 1])

    reconstruction = model(simplified_train[tf.newaxis, :, :]).numpy().squeeze()
    rec = np.array(reconstruction)
    plt.subplot(224)
    plt.title("style")
    plt.xlim(-150, 150)
    plt.ylim(-150, 150)
    for j in range(5):
        plt.plot(rec[j][:, 0], rec[j][:, 1])

    if not os.path.isdir(f'results/train'):
        os.mkdir(f'results/train')
    if not os.path.isdir(f'results/train/{epoch}'):
        os.mkdir(f'results/train/{epoch}')
    plt.savefig(f'results/train/{epoch}/{k}.png')


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

    return tf.keras.backend.sum(tf.sqrt(tf.keras.backend.sum(tf.square(labels - predictions), axis=3)) / 100) / 32


def run(load=0):
    model = MyModel()
    epoch = 200
    if load > 0:
        model.load_weights(f"results/weights/{load}.tf")

    graph_train = []
    graph_test = []
    train_model(model, epoch, loss_object_func, graph_train, graph_test)
    model.summary()
    plt.figure()
    plt.plot(graph_train, label="train")
    plt.plot(graph_test, label="test")
    plt.title("loss vs epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    if not os.path.isdir("results"):
        os.mkdir("results")
    plt.savefig(f'results/{epoch}.png')
    for k in range(50):
        simplified_test = x_test[k]
        label_test = y_test[k]
        visualize_reconstruction(simplified_test, label_test, model, k, epoch)

        # simplified_train = x_train[k]
        # label_train = y_train[k]
        # visualize_train(simplified_train, label_train, model, k, epoch)


# run()