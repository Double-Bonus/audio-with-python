# Basic implementation of UrbanSound8K dataset classification
# For starting achieved ~60% val. acc.

# Based on: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

''' TODO:
   - Properly resize audio samples
   - Add Time Shift (optional?) https://towardsdatascience.com/audio-deep-learning-made-simple-part-3-data-preparation-and-augmentation-24c6e1f6b52
   - Update model structure to get better results
   - Check model fit cb early stoper api
   - use openCV or skimage not both!
'''

# Basic Libraries
import pandas as pd
import numpy as np
from tensorflow.python.keras import callbacks

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

# Libraries for Classification and building Models
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Flatten, Dropout


# Project Specific Libraries
import os
import librosa
import librosa.display
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
    img = 255 - img            # invert. make black==more energy

    # save as PNG
    if not os.path.exists("img_save"):
        os.makedirs("img_save")
    
    skimage.io.imsave(("img_save//" + out), img)
    
    
'''Saves spectograms data from sound files as png pictures'''
def save_wav_to_png():
    for i in range(DATA_SAMPLES_CNT):
        file_name = BASE_PATH  + "//audio//fold" + str(df["fold"][i]) + '//' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        y, sr = librosa.load(file_name, res_type='kaiser_fast') 
        
        out = 'out' + str(i+1) + "_" + str(df["class"][i]) + '.png'
        hop_length = 512           # number of samples per time-step in spectrogram
        n_mels = IMG_HEIGHT        # number of bins in spectrogram. Height of image
        time_steps = IMG_WIDTH - 1 # number of time-steps. Width of image (TODO FIX it add 1 px to width!!)
        
        
        # sr * 4 = size!!!
        # 22050 * 4 = 88200
        y = librosa.util.utils.fix_length(y, 88200)  ## suppose it works????? It just adds white space!!!!
        # y, sr = pad_trunc(y, sr, 4000)
        
        start_sample = 0 # starting at beginning
        length_samples = time_steps * hop_length
        window = y[start_sample:start_sample+length_samples]
        
        spectrogram_image(y=window, sr=sr, out=out, hop_length=hop_length, n_mels=n_mels)

'''Loads images to ram from folder. Also returns class_name for y values'''
def load_spectograms():
    img_data_array = np.zeros((DATA_SAMPLES_CNT, IMG_HEIGHT, IMG_WIDTH)) # some how it adds one pixcel
    class_name = np.zeros((DATA_SAMPLES_CNT, 1))
    
    cla = np.array(df["classID"]) # TODO FIX suppose its global

    for i in range(0, DATA_SAMPLES_CNT):
        image_path = "img_save//" + "out" + str(i+1) + "_" + str(df["class"][i]) + ".png"
        image= cv2.imread(image_path, cv2.COLOR_BGR2RGB) # TODO tikrai toks color map???????
        # image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
        image = np.array(image)
        image = image.astype('float32')
        image /= 255
        img_data_array[i] = image 
        class_name[i] = cla[i]  
    return img_data_array, class_name

def draw_model_results(model_history):
    plt.subplot(121)
    plt.plot(model_history.history['accuracy'], 'r')
    plt.plot(model_history.history['val_accuracy'], 'b')
    plt.ylabel('accuracy, r - train, b - val')
    plt.xlabel('epoch')
    plt.grid(b=True)

    plt.subplot(122)
    plt.plot(model_history.history['loss'], 'r')
    plt.plot(model_history.history['val_loss'], 'b')
    plt.ylabel('Loss, r - train, b - val')
    plt.xlabel('epoch')
    plt.grid(b=True)
    plt.show()

def train_CNN(x_train, y_train, x_test, y_test):
    
    x_train = x_train.reshape(DATA_SAMPLES_CNT - TEST_SAMPLES_CNT, IMG_HEIGHT, IMG_WIDTH, 1)
    x_test = x_test.reshape(TEST_SAMPLES_CNT, IMG_HEIGHT, IMG_WIDTH, 1)
    
    model = Sequential()
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='tanh', input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))     
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu' ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))    
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu' ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))    
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu' ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation = "tanh"))
    model.add(Dense(units=CLASSES_CNT, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    model.summary()

    earlystopper = keras.callbacks.EarlyStopping(patience=7, verbose=1, monitor='val_accuracy')
    checkpointer = keras.callbacks.ModelCheckpoint('models\\urban_model.h5', verbose=1, save_best_only=True)

    hist = model.fit(x_train, y_train, batch_size=64, epochs=80, verbose=1, validation_data=(x_test, y_test), callbacks = [earlystopper, checkpointer])
    draw_model_results(hist)

    
# ----------------------- MAIN ------------------
DEBUG_MODE = 0
BASE_PATH = "Urband_sounds//UrbanSound8K"
DATA_SAMPLES_CNT = 8732
TEST_SAMPLES_CNT = 837  # only 10 fold for testing
CLASSES_CNT = 10
IMG_HEIGHT = 64
IMG_WIDTH = 128

df = pd.read_csv("Urband_sounds//UrbanSound8K//metadata//UrbanSound8K_edit.csv")
if DEBUG_MODE:
    print(df.head())
    
    show_basic_data()
    show_diff_classes()


# suppose existanse of images folder shows that there is data
if not os.path.exists("img_save"):
    save_wav_to_png()

X_data, Y_data = load_spectograms()

# there is tensorflow api for this!!!!!
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

x_train = X_data[0:DATA_SAMPLES_CNT-TEST_SAMPLES_CNT]
x_test = X_data[DATA_SAMPLES_CNT-TEST_SAMPLES_CNT:DATA_SAMPLES_CNT]

y_train = Y_data[0 : DATA_SAMPLES_CNT-TEST_SAMPLES_CNT]
y_test  = Y_data[DATA_SAMPLES_CNT-TEST_SAMPLES_CNT : DATA_SAMPLES_CNT]

print(x_train.shape)
print(np.min(x_train), np.max(x_train))
print(y_train)


train_CNN(x_train, y_train, x_test, y_test)


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Done! at: ", current_time)