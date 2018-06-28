from keras.layers import Input, Flatten, Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization

def OCRModel(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3))(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Conv2D(32, (3, 3))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(64, (3, 3))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    # x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    return predictions, inputs
