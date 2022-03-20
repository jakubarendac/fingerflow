import tensorflow as tf
import numpy as np

from . import constants

# TODO : write tests


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


def verify_net_loss(margin):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - normally used value is 1.

    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

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

        return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return contrastive_loss


def get_input_shape(precision):
    """Provides input shape based on precision

    Arguments:
        precision: number of minutia to be used

    Returns:
        Input shape tuple - number of minutiae, used features
    """
    return (precision, constants.MINUTIAE_FEATURES, 1)


def preprocess_predict_input(anchor, sample):
    """Provides preprocessed predict input

    Arguments:
        anchor: anchor minutiae points
        sample: sample minutiae points

    Returns:
        Array consumed by model.predict function
    """
    return [np.array([anchor]), np.array([sample])]
