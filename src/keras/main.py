import os
import numpy as np
import pickle

from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, BaseLogger, TensorBoard, ReduceLROnPlateau
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from metrics import top_3_acc

from datetime import datetime
from models import OCRModel

DATASET_ROOT_PATH = '/Users/filipgulan/college/ocr/data/dataset_split'
LR = 1e-3
MIN_LR = 1e-7
NUM_CLASSES = 67
BATCH_SIZE = 48
NUM_CHANNELS = 1
INPUT_SIZE = (40, 40)  # h x w
EPOCHS = 10000
NUM_THREADS = 4


def get_callbacks():

    callbacks = list()

    callbacks.append(EarlyStopping(monitor='val_loss', patience=20, verbose=1))

    weights_file = os.path.join(
        "weights", "weights_ep_{epoch:02d}_{val_acc:.5f}.hd5f")
    callbacks.append(ModelCheckpoint(weights_file, monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='max'))

    callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.3, patience=10, min_lr=MIN_LR))

    callbacks.append(BaseLogger())

    callbacks.append(TensorBoard())

    return callbacks


def main():

    predictions, inputs = OCRModel((*INPUT_SIZE, NUM_CHANNELS), NUM_CLASSES)
    # this is the model we will train
    model = Model(inputs=inputs, outputs=predictions)
    print(model.summary())

    # Create data generators
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=10,
                                       zoom_range=(0.7, 1.3),
                                       width_shift_range=0.15,
                                       height_shift_range=0.15)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will and indefinitely
    # generate batches of augmented image data
    # target_size: tuple of integers (height, width)
    train_flow = train_datagen.flow_from_directory(
        os.path.join(DATASET_ROOT_PATH, 'train'),
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale')

    validation_flow = validation_datagen.flow_from_directory(
        os.path.join(DATASET_ROOT_PATH, 'validation'),
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale')

    optimizer = optimizers.Adam(lr=LR)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', top_3_acc])

    # train the model on the new data for a few epochs
    history = model.fit_generator(
        train_flow,
        workers=NUM_THREADS,
        max_queue_size=round(NUM_THREADS * 1.7),
        use_multiprocessing=False,
        steps_per_epoch=train_flow.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=get_callbacks(),
        shuffle=True,
        validation_data=validation_flow,
        validation_steps=validation_flow.samples // BATCH_SIZE)

    with open('./history/hist.pkl', 'wb') as file_pkl:
        pickle.dump(history.history, file_pkl)


if __name__ == "__main__":
    main()
