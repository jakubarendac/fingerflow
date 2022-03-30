# pylint: skip-file
import tensorflow as tf

from . import constants, utils


def get_verify_net_model(precision, verify_net_path=None):
    embedding_network = get_embeddings_model(precision)

    input_1 = tf.keras.Input(utils.get_input_shape(precision))
    # x1 = tf.keras.layers.ZeroPadding2D((0, 7))(input_1)
    # x1 = tf.keras.layers.ZeroPadding2D((22, 22))(x1)

    # x1 = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(x1)

    input_2 = tf.keras.Input(utils.get_input_shape(precision))
    # x2 = tf.keras.layers.ZeroPadding2D((0, 7))(input_1)
    # x2 = tf.keras.layers.ZeroPadding2D((22, 22))(x2)

    # x2 = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(x2)

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
    switcher = {
        15: build_15_minutiae_model,
        20: build_20_minutiae_model
    }

    inputs = tf.keras.Input(shape=(utils.get_input_shape(precision)))

    x = tf.keras.layers.BatchNormalization()(inputs)

    outputs = switcher.get(precision)(x)

    # return outputs
    # # x = tf.keras.layers.Dense(32, activation='relu')(x)
    # # x = tf.keras.layers.Dense(64, activation='sigmoid')(x)
    # # x = tf.keras.layers.Dense(128, activation='relu')(inputs)

    # # x = tf.keras.layers.Dense(128,
    # #                           kernel_regularizer=tf.keras.regularizers.l1(l1=0.001))(x)
    # # x = tf.keras.layers.PReLU()(x)
    # # x = tf.keras.layers.Dropout(0.1)(x)

    # # something good
    # # x = tf.keras.layers.Conv1D(64, 3, activation=tf.keras.layers.PReLU(),
    # #                            kernel_regularizer=tf.keras.regularizers.l2(l2=0))(x)

    # # x = tf.keras.layers.BatchNormalization()(x)
    # # # x = tf.keras.layers.PReLU()(x)
    # # # x = tf.keras.layers.Dropout(0.2)(x)
    # # # x = tf.keras.layers.MaxPooling1D(2)(x)

    # # # x = tf.keras.layers.Conv1D(128, 3)(x)
    # # # # x = tf.keras.layers.PReLU()(x)
    # # # x = tf.keras.layers.Dropout(0.2)(x)
    # # # x = tf.keras.layers.MaxPooling1D(2)(x)

    # # x = tf.keras.layers.Conv1D(64, 3, activation=tf.keras.layers.PReLU(),
    # #                            kernel_regularizer=tf.keras.regularizers.l2(l2=0))(x)

    # # x = tf.keras.layers.BatchNormalization()(x)
    # # # x = tf.keras.layers.PReLU()(x)
    # # # x = tf.keras.layers.Dropout(0.2)(x)
    # # # x = tf.keras.layers.MaxPooling1D(2)(x)

    # # x = tf.keras.layers.Dense(32, activation=tf.keras.layers.PReLU())(x)
    # # something good
    # # x = tf.keras.layers.PReLU()(x)
    # # x = tf.keras.layers.Dropout(0.1)(x)

    # # x = tf.keras.layers.Dense(128,
    # #                           kernel_regularizer=tf.keras.regularizers.l1(l1=0.001))(x)
    # # x = tf.keras.layers.PReLU()(x)
    # # x = tf.keras.layers.Dropout(0.1)(x)

    # # x = tf.keras.layers.Dense(128, activation='sigmoid')(x)
    # # x = tf.keras.layers.Dense(64, activation='sigmoid')(x)

    # x = tf.keras.layers.Dense(64, activation=tf.keras.layers.PReLU())(x)

    # x = tf.keras.layers.Dense(5, activation=tf.keras.layers.PReLU())(x)

    # x = tf.keras.layers.Flatten()(x)

    # # outputs = tf.keras.layers.Dense(5, activation='relu')(x)

    # # x = tf.keras.layers.Conv1D(64, 3)(x)
    # # x = tf.keras.layers.PReLU()(x)
    # # x = tf.keras.layers.MaxPooling1D(2)(x)

    embedding_network = tf.keras.Model(inputs, outputs)
    embedding_network.summary()
    #model = ResNet50()

    return embedding_network


def KerasResNet50():
    base_model = tf.keras.applications.ResNet50(
        weights=None, include_top=False, input_shape=(64, 64, 3),
        classifier_activation="softmax")

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    predictions = tf.keras.layers.Dense(256, activation='softmax')(x)

    # base_model.trainable = False
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    model.summary()

    return model


# def MergeModel(input_shape):
#     X_input = tf.keras.layers.Input(input_shape)

#     # X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)
#     # Zero-Padding
#     X = tf.keras.layers.ZeroPadding2D((0, 7))(X_input)
#     X = tf.keras.layers.ZeroPadding2D((25, 25))(X)

#     X = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(X)

#     preprocessing_model = tf.keras.Model(inputs=X_input, outputs=X)
#     keras_model = KerasModel()

#     concatenated = tf.keras.layers.merge.concatenate([model1_out, model2_out])


def build_15_minutiae_model(x):
    x = tf.keras.layers.Conv1D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv1D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    return x


def build_20_minutiae_model(x):
    # x = tf.keras.layers.Conv1D(64, 3, activation="relu")(x)
    # # x = tf.keras.layers.MaxPooling1D(2)(x)
    # # x = tf.keras.layers.Dropout(0.2)(x)

    # x = tf.keras.layers.Conv1D(128, 3, activation="relu",
    #                           kernel_regularizer=tf.keras.regularizers.l2(l2=0))(x)
    #x = tf.keras.layers.MaxPooling1D(1)(x)
    #x = tf.keras.layers.Dropout(0.2)(x)

    # x = tf.keras.layers.Conv1D(128, 3, activation="relu")(x)

    # x = tf.keras.layers.Conv1D(256, 3, activation="relu")(x)

    # x = tf.keras.layers.Conv1D(128, 3, activation="relu")(x)
    # # x = tf.keras.layers.MaxPooling1D(2)(x)

    # x = tf.keras.layers.Conv1D(16, 15, activation="relu")(x)

    # x = tf.keras.layers.MaxPooling1D(1, strides=2)(x)

    # x = tf.keras.layers.Conv1D(32, 10, strides=5, activation="relu")(x)

    # x = tf.keras.layers.Conv1D(27, 5, strides=1, activation="relu")(x)

    # x = tf.keras.layers.Conv1D(227, 11, strides=4, activation="relu")(x)
    # x = tf.keras.layers.MaxPooling1D(2)(x)
    # x = tf.keras.layers.Dropout(0.2)(x)

    # x = tf.keras.layers.Conv1D(8, 3, activation="relu")(x)

    # x = tf.keras.layers.Conv1D(256, 3, activation="relu",
    #                           kernel_regularizer=tf.keras.regularizers.l2(l2=0))(x)
    #x = tf.keras.layers.MaxPooling1D(1)(x)
    #x = tf.keras.layers.Dropout(0.1)(x)

    # x = tf.keras.layers.Conv1D(512, 3, activation="relu",
    #                           kernel_regularizer=tf.keras.regularizers.l2(l2=0))(x)
    #x = tf.keras.layers.MaxPooling1D(1)(x)
    #x = tf.keras.layers.Dropout(0.1)(x)

    # x = tf.keras.layers.Conv1D(256, 3, activation="relu",
    #                           kernel_regularizer=tf.keras.regularizers.l2(l2=0))(x)
    #x = tf.keras.layers.MaxPooling1D(1)(x)
    #x = tf.keras.layers.Dropout(0.1)(x)

    #x = tf.keras.layers.Conv1D(128, 3, activation="relu")(x)
    #x = tf.keras.layers.MaxPooling1D(1)(x)
    #x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.ZeroPadding2D((0, (6, 5)))(x)
#    X = tf.keras.layers.ZeroPadding2D((25, 25))(X)

    # x = tf.keras.layers.Conv2D(1, (3,3),
    #                           kernel_regularizer=tf.keras.regularizers.l2(l2=0.001),
    #                           activation="relu")(x)
    #x = tf.keras.layers.MaxPooling1D(1)(x)
    #x = tf.keras.layers.Dropout(0.2)(x)

    # x = tf.keras.layers.Dense(1024, activation='sigmoid')(x)

    #x = tf.keras.layers.Dense(256, activation='sigmoid')(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    #x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Conv1D(64, 3, 2, activation="relu")(x)
    #x = tf.keras.layers.MaxPooling2D(2)(x)
    #x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv1D(64, 3, 2, activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    #x = tf.keras.layers.Dropout(0.1)(x)

    # x = tf.keras.layers.Dense(5, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(256,
                              kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001),
                              activation='relu')(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.Dense(128,
                              kernel_regularizer=tf.keras.regularizers.l2(l2=0.0001),
                              activation='relu')(x)
    x = tf.keras.layers.Dropout(0.6)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation='sigmoid')(x)
    # x = tf.keras.layers.Dropout(0.1)(x)

    return x


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1),
                               strides=(1, 1),
                               padding='valid', name=conv_name_base + '2a')(X)
    X = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f, f),
                               strides=(1, 1),
                               padding='same', name=conv_name_base + '2b')(X)
    X = tf.keras.layers.BatchNormalization(name=bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1),
                               strides=(1, 1),
                               padding='valid', name=conv_name_base + '2c')(X)
    X = tf.keras.layers.BatchNormalization(name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = tf.keras.layers.Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a')(X)
    # X = tf.keras.layers.Dropout(0.2)
    X = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = tf.keras.layers.Conv2D(
        filters=F2, kernel_size=(f, f),
        strides=(1, 1),
        padding='same', name=conv_name_base + '2b')(X)
    # X = tf.keras.layers.Dropout(0.2)
    X = tf.keras.layers.BatchNormalization(name=bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = tf.keras.layers.Conv2D(
        filters=F3, kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid', name=conv_name_base + '2c')(X)
    # X = tf.keras.layers.Dropout(0.2)
    X = tf.keras.layers.BatchNormalization(name=bn_name_base + '2c')(X)

    # SHORTCUT PATH #### (≈2 lines)
    X_shortcut = tf.keras.layers.Conv2D(
        filters=F3, kernel_size=(1, 1),
        strides=(s, s),
        padding='valid', name=conv_name_base + '1')(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(
        name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X

# https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Network_Keras.ipynb
# https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33


def ResNet50(input_shape=(20, 6, 1)):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV1D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = tf.keras.layers.Input(input_shape)

    # X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)
    # Zero-Padding
    X = tf.keras.layers.ZeroPadding2D((0, 7))(X_input)
    X = tf.keras.layers.ZeroPadding2D((25, 25))(X)

    X = tf.keras.layers.Conv2D(3, (3, 3), padding='same')(X)

    # Stage 1
    X = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.Dropout(0.1)(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), name='conv2',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
 #   X = tf.keras.layers.Activation('relu')(X)
    # Stage 2
    #X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    #X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    #X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # # ### START CODE HERE ###

    # # # Stage 3 (≈4 lines)
#    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
#    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
#    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
#    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # # # Stage 4 (≈6 lines)
#    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
#    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
#    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
#    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
#    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
#    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # # # Stage 5 (≈3 lines)
#    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
#    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
#    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling1D(...)(X)"
#    X = tf.keras.layers.AveragePooling2D((2, 2), name="avg_pool")(X)

    ### END CODE HERE ###

    # output layer
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(
        16, activation='relu',
        name='fc' + str(16),
        #kernel_regularization = tf.keras.regularizers.l2(l2=0.001),
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    #X = tf.keras.layers.Dropout(0.1)(X)
    # Create model
    model = tf.keras.Model(inputs=X_input, outputs=X, name='ResNet50')
    print("summary => ", model.summary())
    return model
