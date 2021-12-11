# Basic Libraries

import pandas as pd
import numpy as np

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
# matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler



# Libraries for Classification and building Models

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical 

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC




# Project Specific Libraries

import os
import librosa
import librosa.display
import glob 
import skimage


def show_basic_data():
    dat1, sampling_rate1 = librosa.load(BASE_PATH + "//audio//fold5//100032-3-0-0.wav")
    dat2, sampling_rate2 = librosa.load(BASE_PATH + "//audio//fold5//100263-2-0-117.wav")
    plt.figure(figsize=(20, 10))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(dat1)), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    # plt.show()

    # plt.figure(figsize=(20, 10))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(dat2)), ref=np.max)
    plt.subplot(4, 2, 2)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.show()



DEBUG_MODE = 1
BASE_PATH = "Urband_sounds//UrbanSound8K"


df = pd.read_csv("Urband_sounds//UrbanSound8K//metadata//UrbanSound8K.csv")
if DEBUG_MODE:
    print(df.head())
    
    show_basic_data()





'''Using random samples to observe difference in waveforms.'''

arr = np.array(df["slice_file_name"])
fold = np.array(df["fold"])
cla = np.array(df["class"])

j = 1
plt.figure(figsize=(10, 5))
for i in range(175, 197, 3):
    path = BASE_PATH  + "//audio//fold" + str(fold[i]) + '//' + arr[i]
    data, sampling_rate = librosa.load(path)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
    plt.subplot(4, 2, j)
    j = j + 1
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(cla[i])
plt.show()




print('Done!')