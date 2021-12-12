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
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical 

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Project Specific Libraries

import os
import librosa
import librosa.display
import glob 
import skimage.io
import torch

from datetime import datetime
import cv2

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

def show_diff_classes():
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

#------------------ Normal work -----------------------

# Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
def pad_trunc(sig, sr, max_ms):
  num_rows = sig.shape
  sig_len = librosa.get_duration(y=sig, sr=sr)
  max_len = sr//1000 * max_ms
  if (sig_len > max_len):
    # Truncate the signal to the given length
    sig = sig[:,:max_len]
  elif (sig_len < max_len):
    # Length of padding to add at the beginning and end of the signal
    pad_begin_len = np.random.randint(0, max_len - sig_len)
    pad_end_len = max_len - sig_len - pad_begin_len
    # Pad with 0s
    pad_begin = torch.zeros((num_rows, pad_begin_len))
    pad_end = torch.zeros((num_rows, pad_end_len))
    sig = torch.cat((pad_begin, sig, pad_end), 1)
    
  return (sig, sr)


# why ????
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
    
    
    mels = np.log(mels + 1e-9) # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    # save as PNG
    if not os.path.exists("img_save"):
        os.makedirs("img_save")
    
    skimage.io.imsave(("img_save//" + out), img)
    
    
'''Saves data sound files as png pictures.'''
def parser():
    # Function to load files and extract features
    # for i in range(20):
    for i in range(DATA_SAMPLES_CNT):
        file_name = BASE_PATH  + "//audio//fold" + str(df["fold"][i]) + '//' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        y, sr = librosa.load(file_name, res_type='kaiser_fast') 
        
        
        # y, sr = pad_trunc(y, sr, 4000)
        
        out = 'out' + str(i) + '.png' # TODO move out of root dir!!!!!!!!!!!!!!!!
        n_mels = 64 # number of bins in spectrogram. Height of image
        
        
        hop_length = 512 # number of samples per time-step in spectrogram
        n_mels = 128 # number of bins in spectrogram. Height of image
        time_steps = 64 # number of time-steps. Width of image
        
        # window(size) = 
        
        # sr * 4 = size!!!
        # 22050 * 4 = 88200
        y = librosa.util.utils.fix_length(y, 88200)  ## suppose it works????? It just adds white space!!!!
        
        
        start_sample = 0 # starting at beginning
        length_samples = time_steps*hop_length
        window = y[start_sample:start_sample+length_samples]
        
        spectrogram_image(y=window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)

        label.append(df["classID"][i])


def load_spectograms():
    img_data_array = np.zeros((8732, 128, 65))
    # img_data_array = []
    # class_name = []
    class_name = np.zeros((8732, 1))
    
    cla = np.array(df["classID"]) # TODO FIX suppose its global

    for i in range(0, DATA_SAMPLES_CNT):
        image_path = "img_save//" + "out" + str(i) + ".png"
        image= cv2.imread(image_path, cv2.COLOR_BGR2RGB) # TODO tikrai toks color map???????
        # image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
        image = np.array(image)
        image = image.astype('float32')
        image /= 255
        img_data_array[i] = image 
        class_name[i] = cla[i] 
        # img_data_array.append(image)
        #class_name.append(cla[i])   
    return img_data_array, class_name


def train_CNN(x_train, y_train, x_test, y_test):
    
    
    x_train = x_train.reshape(DATA_SAMPLES_CNT - TEST_SAMPLES_CNT, 128, 65, 1)
    x_test = x_test.reshape(TEST_SAMPLES_CNT, 128, 65, 1)
    
    
    
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', input_shape = (128, 65, 1))) # siaip nespalvoti turetu buti!!!!
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu' ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu' ))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu' ))
    # model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu' ))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(units=CLASSES_CNT))
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.summary()

    hist = model.fit(x_train, y_train, batch_size=64, epochs=3, verbose=1, validation_data=(x_test, y_test))
    
    model.save('urban_model.h5')
    
    plt.subplot(121)
    plt.plot(hist.history['accuracy'], 'r')
    plt.plot(hist.history['val_accuracy'], 'b')
    plt.ylabel('accuracy, r - train, b - val')
    plt.xlabel('epoch')
    plt.grid(b=True)

    plt.subplot(122)
    plt.plot(hist.history['loss'], 'r')
    plt.plot(hist.history['val_loss'], 'b')
    plt.ylabel('Loss, r - train, b - val')
    plt.xlabel('epoch')
    plt.grid(b=True)
    plt.show()




DEBUG_MODE = 0
BASE_PATH = "Urband_sounds//UrbanSound8K"
DATA_SAMPLES_CNT = 8732
CLASSES_CNT = 10
TEST_SAMPLES_CNT = 2000

df = pd.read_csv("Urband_sounds//UrbanSound8K//metadata//UrbanSound8K.csv")
if DEBUG_MODE:
    print(df.head())
    
    show_basic_data()
    show_diff_classes()

# store y value to cnn
label = []

# suppose existanse of images folder shows that there is data
if not os.path.exists("img_save"):
    parser()

X_data, Y_data = load_spectograms()
# print(X_data[1].shape) # (128, 65)

# there is tensorflow api for this!!!!!
x_train = X_data[0:DATA_SAMPLES_CNT-TEST_SAMPLES_CNT]
x_test = X_data[DATA_SAMPLES_CNT-TEST_SAMPLES_CNT:DATA_SAMPLES_CNT]

y_train = Y_data[0 : DATA_SAMPLES_CNT-TEST_SAMPLES_CNT]
y_test  = Y_data[DATA_SAMPLES_CNT-TEST_SAMPLES_CNT : DATA_SAMPLES_CNT]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print(x_train.shape)
print(np.min(x_train), np.max(x_train))
print(y_train)


train_CNN(x_train, y_train, x_test, y_test)


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Done! at: ", current_time)