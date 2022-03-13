import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sn
import matplotlib
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

DATASET_PATH = '/home/jakub/projects/dp/matcher_training_data/dataset_20_items/'
INPUT_SHAPE = (20, 6)
LOGS_FOLDER = 'logs-conv-contrast-20/scalars/' + datetime.now().strftime("%Y%m%d-%H%M%S")
EPOCHS = 250
BATCH_SIZE = 16
MARGIN = 1
MODEL_PATH = '/home/jakub/projects/dp/matcher_checkpoints/matcher_contrast_weights_20.h5'


def show_correlation_matrix(data, labels):
    matplotlib.use('TkAgg')
    print(data.reshape(10230, 6).shape)
    dataframe = pd.DataFrame.from_records(data.reshape(10230, 6), columns=labels)

    correlation = dataframe.corr()

    sn.heatmap(correlation, annot=True)
    matplotlib.pyplot.show()


def preprocess_item(filename, folder):
    item = np.genfromtxt(
        f"{folder}/{filename}", delimiter=",")

    return item
    # return MinMaxScaler().fit_transform(item)


def load_folder_data(folder):
    data = []
    labels = []

    for _, _, files in os.walk(folder):
        raw_data = [preprocess_item(filename, folder) for filename in files]

        labels = [int(filename.split("_")[0]) for filename in files]

        data = np.stack(raw_data)

    data_shuffled, labels_shuffled = shuffle(data, np.array(labels))

    return data_shuffled, labels_shuffled


def load_dataset():
    print('loading data')
    data, labels = load_folder_data(DATASET_PATH)

    # show_correlation_matrix(data, ['x', 'y', 'angle', 'score', 'class', 'core_distance'])
    # print("showed matrix")


# load_dataset()
    numClasses = np.max(labels) + 1
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

    # for i in range(0, numClasses):
    #     idxs = np.where(labels == i)[0]
    #     print("{}: {} {}".format(i, len(idxs), idxs))

    pairImages = []
    pairLabels = []

    for idxA in range(len(data)):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = data[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idxB = np.random.choice(idx[label])
        posData = data[idxB]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pairImages.append([currentImage, posData])
        pairLabels.append([1])

        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        negIdx = np.where(labels != label)[0]
        negData = data[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        pairImages.append([currentImage, negData])
        pairLabels.append([0])

        # return a 2-tuple of our image pairs and labels
    return (np.array(pairImages), np.array(pairLabels).astype('float32'))


def split_dataset(pairs, labels):
    length, _ = labels.shape

    train_indices = 0, int(length * 0.6)
    val_indices = int(length * 0.6) + 1, int(length * 0.8)
    test_indices = int(length * 0.8) + 1, length

    train_dataset = (pairs[:train_indices[1]], labels[:train_indices[1]])
    val_dataset = (pairs[val_indices[0]:val_indices[1]], labels[val_indices[0]:val_indices[1]])
    test_dataset = (pairs[test_indices[0]:test_indices[1]], labels[test_indices[0]:test_indices[1]])

    return train_dataset, val_dataset, test_dataset


pairs, labels = load_dataset()
train_dataset, val_dataset, test_dataset = split_dataset(pairs, labels)

# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))


def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

# Identity Block or Residual Block or simply Skip Connector


# def residual_block(X, num_filters: int, stride: int = 1, kernel_size: int = 3,
#                    activation: str = 'relu', bn: bool = True, conv_first: bool = True):
#     """
#     Parameters
#     ----------
#     X : Tensor layer
#         Input tensor from previous layer
#     num_filters : int
#         Conv2d number of filters
#     stride : int by default 1
#         Stride square dimension
#     kernel_size : int by default 3
#         COnv2D square kernel dimensions
#     activation: str by default 'relu'
#         Activation function to used
#     bn: bool by default True
#         To use BatchNormalization
#     conv_first : bool by default True
#         conv-bn-activation (True) or bn-activation-conv (False)
#     """
#     conv_layer = tf.keras.layers.Conv1D(num_filters,
#                                         kernel_size=kernel_size,
#                                         strides=stride,
#                                         padding='same',
#                                         kernel_regularizer=tf.keras.regularizers.l2(1e-4))
#     # X = input
#     if conv_first:
#         X = conv_layer(X)
#         if bn:
#             X = tf.keras.layers.BatchNormalization()(X)
#         if activation is not None:
#             X = tf.keras.layers.Activation(activation)(X)
#             X = tf.keras.layers.Dropout(0.2)(X)
#     else:
#         if bn:
#             X = tf.keras.layers.BatchNormalization()(X)
#         if activation is not None:
#             X = tf.keras.layers.Activation(activation)(X)
#         X = conv_layer(X)

#     return X


inputs = tf.keras.Input(shape=(INPUT_SHAPE))
x = tf.keras.layers.BatchNormalization()(inputs)

x = tf.keras.layers.Conv1D(32, 3, activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l2(l2=0.001))(x)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Conv1D(64, 3, activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l2(l2=0.001))(x)
x = tf.keras.layers.MaxPooling1D(1)(x)
x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Conv1D(64, 3, activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l2(l2=0.001))(x)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Dropout(0.2)(x)

x = tf.keras.layers.Flatten()(x)

# x = tf.keras.regularizers.l1(l1=0.01)(x)
# x = tf.keras.regularizers.l2(l2=0.01)(x)
# x = tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)(x)
# x = tf.keras.layers.AveragePooling1D(2)(x)
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Conv1D(32, 3, activation="relu")(x)
# x = tf.keras.layers.MaxPooling1D(2)(x)
# x = tf.keras.layers.Dropout(0.5)(x)
# depth = 56
# num_filters_in = 32
# num_res_block = int((depth - 2) / 9)

# inputs = tf.keras.Input(shape=(INPUT_SHAPE))

# X = residual_block(X=inputs, num_filters=num_filters_in, conv_first=True)

# # Building stack of residual units
# for stage in range(3):
#     num_filters_out = 32
#     for unit_res_block in range(num_res_block):
#         activation = 'relu'
#         bn = True
#         stride = 1
#         # First layer and first stage
#         if stage == 0:
#             num_filters_out = num_filters_in * 4
#             if unit_res_block == 0:
#                 activation = None
#                 bn = False
#             # First layer but not first stage
#         else:
#             num_filters_out = num_filters_in * 2
#             if unit_res_block == 0:
#                 stride = 2

#         # bottleneck residual unit
#         y = residual_block(X,
#                            num_filters=num_filters_in,
#                            kernel_size=1,
#                            stride=stride,
#                            activation=activation,
#                            bn=bn,
#                            conv_first=False)
#         y = residual_block(y,
#                            num_filters=num_filters_in,
#                            conv_first=False)
#         y = residual_block(y,
#                            num_filters=num_filters_out,
#                            kernel_size=1,
#                            conv_first=False)
#         if unit_res_block == 0:
#             # linear projection residual shortcut connection to match
#             # changed dims
#             X = residual_block(X=X,
#                                num_filters=num_filters_out,
#                                kernel_size=1,
#                                stride=stride,
#                                activation=None,
#                                bn=False)
#         X = tf.keras.layers.add([X, y])
#     num_filters_in = num_filters_out

# # normalization_layer = tf.keras.layers.BatchNormalization()(inputs)
# conv1 = tf.keras.layers.Conv1D(128, 3, activation='relu',
#                                input_shape=(INPUT_SHAPE))(x)
# conv1drop = tf.keras.layers.Dropout(0.5)(conv1)
# pool1 = tf.keras.layers.MaxPooling1D(2)(conv1drop)
# conv2 = tf.keras.layers.Conv1D(256, 3, activation='relu')(pool1)
# conv2drop = tf.keras.layers.Dropout(0.5)(conv2)
# # # pool2 = tf.keras.layers.MaxPooling1D(2)(conv2drop)
# # conv3 = tf.keras.layers.Conv1D(128, 3, activation='relu')(conv2drop)
# # conv3drop = tf.keras.layers.Dropout(0.5)(conv3)
# # pool3 = tf.keras.layers.MaxPooling1D(2)(conv3drop)
# # conv4 = tf.keras.layers.Conv1D(256, 3, activation='relu')(pool3)
# # # conv4drop = tf.keras.layers.Dropout(0.5)(conv4)
# # pool4 = tf.keras.layers.MaxPooling1D(3)(conv4)
# x = tf.keras.layers.Flatten()(conv2drop)


# # x = tf.keras.layers.BatchNormalization()(X)
# # x = tf.keras.layers.Dense(10, activation="relu")(x)

# X = tf.keras.layers.BatchNormalization()(X)
# X = tf.keras.layers.Activation('relu')(X)
# X = tf.keras.layers.AveragePooling1D(pool_size=2)(X)
# y = tf.keras.layers.Flatten()(X)
# y = tf.keras.layers.Dense(512, activation='relu')(y)
# y = tf.keras.layers.BatchNormalization()(y)
# y = tf.keras.layers.Dropout(0.5)(y)

outputs = tf.keras.layers.Dense(5,
                                activation='sigmoid')(x)
embedding_network = tf.keras.Model(inputs, outputs)


input_1 = tf.keras.Input(INPUT_SHAPE)
input_2 = tf.keras.Input(INPUT_SHAPE)

# As mentioned above, Siamese Network share weights between
# tower networks (sister networks). To allow this, we will use
# same embedding network for both tower networks.
tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = tf.keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = tf.keras.Model(inputs=[input_1, input_2], outputs=output_layer)


def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

  Arguments:
      margin: Integer, defines the baseline for distance for which pairs
              should be classified as dissimilar. - (default is 1).

  Returns:
      'constrastive_loss' function with data ('margin') attached.
  """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.

      Returns:
          A tensor containing constrastive loss as floating point value.
      """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


# def lr_schedule(epoch):
#     """Learning Rate Schedule
#     """
#     l_r = 0.5e-2
#     if epoch > 180:
#         l_r *= 0.5e-3
#     elif epoch > 150:
#         l_r *= 1e-3
#     elif epoch > 60:
#         l_r *= 5e-2
#     elif epoch > 30:
#         l_r *= 5e-1
#     print(('Learning rate: ', l_r))
#     return l_r


tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOGS_FOLDER)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH, monitor=['val_accuracy'],
    verbose=1, mode='max', save_weights_only=True)
learning_rate = tf.keras.callbacks.ReduceLROnPlateau(verbose=1, patience=20)
# # siamese.compile(loss="binary_crossentropy", optimizer="RMSprop", metrics=["accuracy"])
siamese.compile(
    loss=loss(margin=MARGIN),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=["accuracy"])
siamese.summary()

train_pairs, train_labels = train_dataset
val_pairs, val_labels = val_dataset
test_pairs, test_labels = test_dataset

history = siamese.fit(
    [train_pairs[:, 0], train_pairs[:, 1]],
    train_labels,
    validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[tensorboard, checkpoint]
)

predictions = siamese.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_labels)
