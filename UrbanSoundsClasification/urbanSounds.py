# Basic implementation of UrbanSound8K dataset classification
# Using 75/25 train/test spilt achieved ~90-92% val. acc. (TODO for proper validation need
#   to make: "Use the predefined 10 folds and perform 10-fold (not 5-fold) cross validation")
# To get better result could try to combine multiple spectral features, for exemple Mel spectram, chromagram or tonnetz.

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


# Libraries for Classification and building Models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization, ZeroPadding2D
from tensorflow.keras import callbacks

# Project Specific Libraries
import os
import librosa
import librosa.display
import skimage.io
import torch
from datetime import datetime
import cv2
import sklearn

# Packets TODO it's how they are called???
from visualise import log_confusion_matrix, show_basic_data, show_diff_classes, draw_model_results, show_mel_img
from visualise import plot_wave_from_audio

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

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def spectrogram_image(y, sr, out_path, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
    # mels = librosa.feature.melspectrogram(y=y, sr=sr)
    
    if 1:
        mels = np.log(mels + 1e-9) # add small number to avoid log(0)
    else:  #testing !
        mels = np.mean(mels, axis=0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255 - img            # invert. make black==more energy

    # save as PNG
    if not os.path.exists("img_save"):
        os.makedirs("img_save")
    
    cv2.imwrite(("img_save//" + out_path), img)
    
    
'''Saves spectograms data from sound files as png pictures'''
def save_wav_to_png():
    print("Saving pictures to drive")
    for i in range(DATA_SAMPLES_CNT):
        file_name = BASE_PATH  + "//audio//fold" + str(df["fold"][i]) + '//' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        y, sr = librosa.load(file_name, res_type='kaiser_fast') 
        
        img_path = 'out' + str(i+1) + "_" + str(df["class"][i]) + '.png'
        hop_length = 512           # number of samples per time-step in spectrogram
        n_mels = IMG_HEIGHT        # number of bins in spectrogram. Height of image
        time_steps = IMG_WIDTH - 1 # number of time-steps. Width of image (TODO FIX it add 1 px to width!!)
        
        
        # sr * 4 = size!!!
        # 22050 * 4 = 88200
        # y = librosa.util.utils.fix_length(y, 75000)  ## suppose it works????? It just adds white space!!!!
        y = librosa.util.utils.fix_length(y, 88200)  ## suppose it works????? It just adds white space!!!!
        # y, sr = pad_trunc(y, sr, 4000)
        
        start_sample = 0 # starting at beginning
        length_samples = time_steps * hop_length
        window = y[start_sample:start_sample+length_samples]
        
        spectrogram_image(y=window, sr=sr, out_path=img_path, hop_length=hop_length, n_mels=n_mels)
    print("Done saving pictures!")
    

'''Loads images to ram from folder. Also returns class_name for y values'''
def load_spectograms():
    img_data_array = np.zeros((DATA_SAMPLES_CNT, IMG_HEIGHT, IMG_WIDTH)) # some how it adds one pixcel
    class_name = np.zeros((DATA_SAMPLES_CNT, 1))
    
    cla = np.array(df["classID"]) # TODO FIX suppose its global

    for i in range(0, DATA_SAMPLES_CNT):
        image_path = "img_save//" + "out" + str(i+1) + "_" + str(df["class"][i]) + ".png"
        image= cv2.imread(image_path, cv2.COLOR_BGR2RGB) # TODO tikrai toks color map???????
        # image= cv2.imread(image_path)
        if image is None:
            print("Error, image was not found from: " + image_path)
            quit()
        # image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
        image = np.array(image)
        image = image.astype('float32')
        image /= 255
        img_data_array[i] = image 
        class_name[i] = cla[i]  
    return img_data_array, class_name


def train_CNN(X, Y, test_portion = 0.25):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=test_portion, random_state=7)
    
    x_train = x_train.reshape(x_train.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
    
    train_labels = keras.utils.to_categorical(y_train, num_classes=CLASSES_CNT)
    test_labels = keras.utils.to_categorical(y_test, num_classes=CLASSES_CNT)
    
    # Initialize model
    model = Sequential()
    
    # Layer 1
    model.add(Conv2D(filters=96, kernel_size=(3,3), activation='relu', input_shape = (IMG_HEIGHT, IMG_WIDTH, 1), padding='same'))
    model.add(MaxPooling2D((2, 2)))
     
    # Layer 2
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same' ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))    
    
    # Layer 3
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same' ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))    
    
    # Layer 4
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same' ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    
    # Layer 5
    model.add(Flatten())
    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(0.5))
    
    # Layer 6
    model.add(Dense(CLASSES_CNT, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    model.summary()

    earlystopper = callbacks.EarlyStopping(patience=7, verbose=1, monitor='accuracy')
    checkpointer = callbacks.ModelCheckpoint('models\\urban_model.h5', verbose=1, save_best_only=True)
    
    hist = model.fit(x_train, train_labels, batch_size=128, epochs=40, verbose=1, validation_data=(x_test, test_labels), callbacks = [earlystopper, checkpointer])
    draw_model_results(hist)
    log_confusion_matrix(model, x_test, y_test) # Note that here you use last model not the one saved!
    
    
# ----------------------- MAIN ------------------
DEBUG_MODE = 0
BASE_PATH = "Urband_sounds//UrbanSound8K"
DATA_SAMPLES_CNT = 8732
CLASSES_CNT = 10
TEST_PORTION = 0.25
IMG_HEIGHT = 64
IMG_WIDTH = 128

df = pd.read_csv("Urband_sounds//UrbanSound8K//metadata//UrbanSound8K.csv")
if DEBUG_MODE:
    print(df.head())
    show_basic_data(BASE_PATH)
    show_diff_classes(df, BASE_PATH)
    show_mel_img(BASE_PATH, IMG_HEIGHT)
    plot_wave_from_audio(df, BASE_PATH)
    
# suppose existanse of images folder shows that there is data
if not os.path.exists("img_save"):
    save_wav_to_png()

X_data, Y_data = load_spectograms()


train_CNN(X_data, Y_data, TEST_PORTION)


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Done! at: ", current_time)