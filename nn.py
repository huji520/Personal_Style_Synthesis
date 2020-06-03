import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Add, Activation, LeakyReLU
from tensorflow.keras import Model
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# simplify_clusters_shape = pickle.load(open('x/40_1_0.5_new.p', "rb"))
# person_clusters_shape = pickle.load(open('y/40_1_0.5_new.p', "rb"))

x_train = pickle.load(open('x/40_0_0.5_train_new.p', "rb"))
y_train_0 = pickle.load(open('x/40_0_0.5_train_new_0.p', "rb"))
y_train_1 = pickle.load(open('x/40_0_0.5_train_new_1.p', "rb"))
y_train = []
y_train.extend(y_train_0)
y_train.extend(y_train_1)
x_test = pickle.load(open('x/40_0_0.5_test_new.p', "rb"))
y_test = pickle.load(open('x/40_0_0.5_test_new.p', "rb"))

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
#     plt.savefig(f'results/input/test/{i}.png')


# indexes1 = np.arange(len(simplify_clusters_shape))
# np.random.shuffle(indexes1)
#
# simplify_clusters_shape = simplify_clusters_shape[indexes1]
# person_clusters_shape = person_clusters_shape[indexes1]

# x_train = simplify_clusters_shape[:-200]
# y_train = person_clusters_shape[:-200]
#
# x_test = simplify_clusters_shape[-200:]
# y_test = person_clusters_shape[-200:]

x_train = x_train[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(16)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(16)

print(x_train.shape)
print(y_train.shape)


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.reshape = Reshape((35, 30, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(60, activation='relu')
        self.dense2 = Dense(120, activation='relu')
        self.dense2 = Dense(240, activation='relu')
        self.dense3 = Dense(480, activation='relu')
        self.dense4 = Dense(960, activation='relu')
        self.dense5 = Dense(1920, activation='relu')
        self.dense6 = Dense(2100)

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
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

    i = 0
    j = 0.0

    for stroke in labels[0]:
        if tf.math.count_nonzero(stroke) == 0:
            break
        i += 1
        j += 1

    a = tf.keras.backend.sum(
        tf.sqrt(tf.keras.backend.sum(tf.square(labels[0][:i] - predictions[0][:i]), axis=2))) / (j * 30)

    k = 0
    for label in labels:
        if k == 0:
            k += 1
            continue
        i = 0
        j = 0.0
        for stroke in label:
            if tf.math.count_nonzero(stroke) == 0:
                break
            i += 1
            j += 1

        a += tf.keras.backend.sum(
            tf.sqrt(tf.keras.backend.sum(tf.square(labels[k][:i] - predictions[k][:i]), axis=2))) / (j*30)
        k += 1

    return a / 16
    # tf.print("predictions", predictions[0][:1,:3,:])
    # tf.print("labels", labels[0][:1,:3,:])
    # tf.print("diff", labels[0][:1, :3, :] - predictions[0][:1, :3, :])
    # tf.print("square", tf.square(labels[0][:1, :3, :] - predictions[0][:1, :3, :]))
    # tf.print("sum+square", tf.keras.backend.sum(tf.square(labels[0][:1, :3, :] - predictions[0][:1, :3, :]), axis=2))
    # tf.print("sum+square+sqrt",
    #          tf.sqrt(tf.keras.backend.sum(tf.square(labels[0][:1, :3, :] - predictions[0][:1, :3, :]), axis=2)))
    # tf.print("loss", tf.keras.backend.sum(tf.sqrt(tf.keras.backend.sum(tf.square(labels[0][:1, :3, :] - predictions[0][:1, :3, :]), axis=2))))

    # tf.print(tf.keras.backend.sum(tf.sqrt(tf.keras.backend.sum(tf.square(labels[0][:i] - predictions[0][:i]), axis=2)) / (j*30)))

    # a = tf.keras.backend.sum(
    #     tf.sqrt(tf.keras.backend.sum(tf.square(labels[0][:i] - predictions[0][:i]), axis=2)) / (i*30))

    # b = tf.keras.backend.sum(
    #     tf.sqrt(tf.keras.backend.sum(tf.square(labels[0][:i] - predictions[0][:i]), axis=2)) / (i*30))
    #
    # c = a+b

    # return a

    # return tf.keras.backend.sum(
    #     tf.sqrt(tf.keras.backend.sum(tf.square(labels[0][:i] - predictions[0][:i]), axis=3)) / (i*30))


    # return tf.keras.backend.sum(tf.sqrt(tf.keras.backend.sum(tf.square(labels - predictions), axis=3)) / 1200) / 32


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


run()