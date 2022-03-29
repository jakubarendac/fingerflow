import tensorflow as tf
import numpy as np

from . import constants


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
    """Provides input shape based on precision.

    Arguments:
        precision: number of minutia to be used.

    Returns:
        Input shape tuple - number of minutiae, used features.
    """
    return (precision, constants.MINUTIAE_FEATURES, 1)


def find_n_nearest_minutiae(minutiae_points, current_minutia):
    def calculate_distance(item):
        return np.linalg.norm(current_minutia-item)

    minutiae_distances = list(map(calculate_distance, minutiae_points))
    minutiae_distances.sort()

    return np.array(minutiae_distances[1:constants.MINUTIA_NEIGHBORS+1])


# TODO : try to add n-nearest not sorted
def enhance_minutiae_points(minutiae):
    """Enhances minutiae points with distances to n-nearest neighbors.
       Also removes `x` an `y` coordinates from feature vector.

    Arguments:
        minutiae: raw extracted minutiae points.

    Returns:
        Numpy array without `x` and `y` with additional distances to n nearest neighbors.
    """
    enhanced_minutiae = []

    for minutia in minutiae:
        features_to_add = find_n_nearest_minutiae(minutiae[:, :2], minutia[:2])

        updated_minutia = np.append(minutia, features_to_add)

        enhanced_minutiae.append(updated_minutia[2:])

    return np.array(enhanced_minutiae)


def preprocess_predict_input(anchor, sample):
    """Provides preprocessed predict input.

    Arguments:
        anchor: anchor minutiae points
        sample: sample minutiae points

    Returns:
        Array consumed by model.predict function
    """
    return [np.array([enhance_minutiae_points(anchor)]), np.array([enhance_minutiae_points(sample)])]
