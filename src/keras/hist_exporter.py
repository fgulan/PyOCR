import os
import pickle
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def load_history():
    pickle_file = open("./history/hist.pkl",'rb')
    history = pickle.load(pickle_file)
    pickle_file.close()
    return history

def export_csv(values, csv_path):
    # values = np.array(values)
    # np.savetxt(csv_path, values, delimiter=",")
    # # np.asarray(values).tofile(csv_path)

    df = pd.DataFrame(values)
    df.to_csv(csv_path, header=None, index=None, float_format='%.10f')

history = load_history()
print(history.keys())
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.ylabel('Točnost')
plt.xlabel('Epoha')
plt.legend(['Skup za učenje', 'Skup za provjeru'], loc='lower right')
plt.savefig("acc.pdf", format='pdf')
plt.show()



plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.ylabel('Gubitak')
plt.xlabel('Epoha')
plt.legend(['Skup za učenje', 'Skup za provjeru'], loc='upper right')
plt.savefig("loss.pdf", format='pdf')
plt.show()

plt.plot(history['lr'])
plt.ylabel('Stopa učenja')
plt.xlabel('Epoha')
plt.savefig("lr.pdf", format='pdf')
plt.show()