import pickle

import matplotlib.pyplot as plt
import numpy as np



def graph_results(hist):
    plt.figure()
    plt.title('Loss in ANN')
    plt.xlabel('Epoch')
    plt.ylabel('val_loss')
    plt.plot(np.array(history['loss']),
             label='Train Loss')
    plt.plot(np.array(history['val_loss']),
             label='Val Loss')
    plt.legend()
    plt.show()


history = pickle.load(open("history.pkl", "rb"))

print(history.keys())

graph_results(history)