import argparse
import numpy as np

from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix

from models import OCRModel

NUM_CLASSES = 67
BATCH_SIZE = 64
NUM_CHANNELS = 1
INPUT_SIZE = (40, 40)

def load_model(weights_path):

    predictions, inputs = OCRModel((*INPUT_SIZE, NUM_CHANNELS), NUM_CLASSES)
    model = Model(inputs=inputs, outputs=predictions)
    model.load_weights(weights_path)

    return model

def load_test_dataset_generator(dataset_path):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_flow = test_datagen.flow_from_directory(
        dataset_path,
        target_size=INPUT_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False)
    
    return test_flow
    
def evaluate(args):
    model = load_model(args.weights)
    test_flow = load_test_dataset_generator(args.dataset)
    import pdb; pdb.set_trace()
    y_pred = model.predict_generator(test_flow, test_flow.samples // BATCH_SIZE + 1)
    y_pred = np.argmax(y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(test_flow.classes, y_pred))
    print('Classification Report')
    print(classification_report(test_flow.classes, y_pred))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights', help="Keras model weights", required=True, type=str)
    parser.add_argument(
        '--dataset', help="Dataset folder", required=True, type=str)
    args = parser.parse_args()
    evaluate(args)

if __name__ == "__main__":
    main()