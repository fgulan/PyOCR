from keras.metrics import top_k_categorical_accuracy


def top_3_acc(x, y, top_k=3):
    return top_k_categorical_accuracy(x, y, k=top_k)
