import argparse
import numpy as np

from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from models import OCRModel
from char_mapper import classifier_out_to_vocab_letter

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
    

def print_global_stats(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro precision:", precision_score(y_true, y_pred, average='macro'))
    print("Macro recall:", recall_score(y_true, y_pred, average='macro'))
    print("Macro f1:", f1_score(y_true, y_pred, average='macro'))


def plot_wrong_classification(x, y_true, y_pred, filename):
    samples = []
    fig_size = (10, 10)
    for i in range(0, x.shape[0]):
        if y_true[i] != y_pred[i]:
            samples.append((y_true[i], y_pred[i], i))

    def plot_sample(x, axis):
        img = x.reshape(x.shape[0], x.shape[1])
        axis.imshow(img, cmap='gray')

    fig = plt.figure(figsize=fig_size)

    for i in range(len(samples)):
        y_t, y_p, index = samples[i]
        ax = fig.add_subplot(*fig_size, i + 1, xticks=[], yticks=[])
        # title = map_int_to_letter(y_t) + " -> " + map_int_to_letter(y_p) 
        ax.title.set_text("title")
        ax.title.set_fontsize(10)
        plot_sample(x[index], ax)

    fig.tight_layout()
    plt.savefig(filename)

def print_wrong_files(files, y_true, y_pred):
    samples = []
    for i in range(0, y_true.shape[0]):
        if y_true[i] != y_pred[i]:
            true_letter = classifier_out_to_vocab_letter(y_true[i])
            pred_letter = classifier_out_to_vocab_letter(y_pred[i])
            filename = files[i]
            sample = "{0}, true: {1}, pred: {2}".format(filename, true_letter, pred_letter)
            samples.append(sample)
    print("\n".join(samples))

def save_confusion_matrix_csv(confusion_matrix, filename):
    num_classes = confusion_matrix.shape[0]
    header = [classifier_out_to_vocab_letter(letter_int) for letter_int in range(num_classes)]
    header = ";".join(header)
    np.savetxt(filename, confusion_matrix.astype(int), fmt='%i', delimiter=";")

import pandas as pd

def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        words = line.split(" ")
        words = list(map(lambda text: text.strip(), words))
        row_data = list(filter(None, words))

        row['Klasa'] = classifier_out_to_vocab_letter(int(row_data[0]))
        row['Preciznost'] = float(row_data[1])
        row['Odziv'] = float(row_data[2])
        row['F1'] = float(row_data[3])
        row['Broj primjera'] = int(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report.csv', index = False)



def evaluate(args):
    model = load_model(args.weights)
    test_flow = load_test_dataset_generator(args.dataset)
    y_pred = model.predict_generator(test_flow, test_flow.samples // BATCH_SIZE + 1)
    y_pred = np.argmax(y_pred, axis=1)
    print_global_stats(test_flow.classes, y_pred)
    print('Confusion Matrix')
    conf_m = confusion_matrix(test_flow.classes, y_pred)
    print(conf_m)
    print('Classification Report')
    print(classification_report(test_flow.classes, y_pred))
    classifaction_report_csv(classification_report(test_flow.classes, y_pred))
    print_wrong_files(test_flow.filenames, test_flow.classes, y_pred)

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