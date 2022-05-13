import tensorflow as tf

from . import constants, utils


def get_verify_net_model(precision, verify_net_path=None):
    embedding_network = get_embeddings_model(precision)

    input_1 = tf.keras.Input(utils.get_input_shape(precision))
    input_2 = tf.keras.Input(utils.get_input_shape(precision))

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = tf.keras.layers.Lambda(utils.euclidean_distance)([tower_1, tower_2])

    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(normal_layer)

    siamese_network = tf.keras.Model(inputs=[input_1, input_2], outputs=output_layer)

    siamese_network.compile(
        loss=utils.verify_net_loss(constants.MARGIN),
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=["accuracy"])

    if verify_net_path:
        siamese_network.load_weights(verify_net_path)

        print(f'Verify net weights loaded from {verify_net_path}')

    return siamese_network


def get_embeddings_model(precision):
    inputs = tf.keras.Input(shape=(utils.get_input_shape(precision)))

    x = tf.keras.layers.Conv2D(32, 3, 2, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(l2=0.001))(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(32, 3, 2, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(l2=0.001))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Flatten()(x)

    outputs = tf.keras.layers.Dense(16, activation="relu")(x)

    embedding_network = tf.keras.Model(inputs, outputs)

    embedding_network.summary()

    return embedding_network
