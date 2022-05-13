import os
import numpy as np


def preprocess_item(filename, folder):
    return np.genfromtxt(
        f"{folder}/{filename}", delimiter=",")


def load_folder_data(folder):
    data = []
    labels = []

    for _, _, files in os.walk(folder):
        raw_data = [preprocess_item(filename, folder) for filename in files]

        labels = [int(filename.split("_")[0]) for filename in files]

        data = np.stack(raw_data)

    return data, np.array(labels)


def load_dataset(dataset_path, positive=True, negative=True):
    print('START: loading data')
    data, labels = load_folder_data(dataset_path)

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

        if positive:
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

        if negative:
            negIdx = np.where(labels != label)[0]

            negData = data[np.random.choice(negIdx)]
            # prepare a negative pair of images and update our lists
            np.random.shuffle(currentImage)
            pairImages.append([currentImage, negData])
            pairLabels.append([0])

        # return a 2-tuple of our image pairs and labels
    print('FINISH: loading data')

    return (np.array(pairImages), np.array(pairLabels).astype('float32'))
