import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

from . import verify_net_model, utils

PRECISION = 20

CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

DATASET_PATH = f'/home/jakub/projects/dp/matcher_training_data/preprocessed_dataset_{PRECISION}'
LOGS_FOLDER = f'/home/jakub/projects/dp/fingerflow/logs/logs-conv-contrast-{PRECISION}/scalars/{CURRENT_TIMESTAMP}'
MODEL_PATH = f'/home/jakub/projects/dp/fingerflow/models/matcher_contrast_weights_{PRECISION}_{CURRENT_TIMESTAMP}.h5'
EPOCHS = 100
BATCH_SIZE = 64

model = verify_net_model.get_verify_net_model(PRECISION)


def preprocess_item(filename, folder):
    raw_minutiae = np.genfromtxt(
        f"{folder}/{filename}", delimiter=",")

    enhanced_minutiae = utils.enhance_minutiae_points(raw_minutiae)

    return enhanced_minutiae


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
    print('START: loading data')
    data, labels = load_folder_data(DATASET_PATH)

    numClasses = np.max(labels) + 1
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

    pairImages = []
    pairLabels = []

    for idxA, _ in enumerate(data):
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
        np.random.shuffle(currentImage)
        pairImages.append([currentImage, posData])
        pairLabels.append([1])

        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        # print(labels)
        negIdx = np.where(labels != label)[0]

        negData = data[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        np.random.shuffle(currentImage)
        pairImages.append([currentImage, negData])
        pairLabels.append([0])

        # return a 2-tuple of our image pairs and labels
    print('FINISH: loading data')

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


def load_and_preprocess_dataset():
    pairs, labels = load_dataset()
    pairs_shuffled, labels_shuffled = shuffle(pairs, labels)
    train_dataset, val_dataset, test_dataset = split_dataset(pairs_shuffled, labels_shuffled)

    return train_dataset, val_dataset, test_dataset


def train():
    train_dataset, val_dataset, test_dataset = load_and_preprocess_dataset()

    model.summary()

    def scheduler(epoch, lr):
        if epoch < 55:
            return lr

        return lr * tf.math.exp(-0.1)

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOGS_FOLDER)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor=['val_accuracy'],
        verbose=1, mode='max', save_weights_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.75,
                                                     patience=10, min_lr=0.0001, verbose=1)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    train_pairs, train_labels = train_dataset
    val_pairs, val_labels = val_dataset
    test_pairs, test_labels = test_dataset

    model.fit(
        [train_pairs[:, 0], train_pairs[:, 1]],
        train_labels,
        validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[tensorboard, checkpoint, reduce_lr, lr_scheduler]
    )

    model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_labels)
