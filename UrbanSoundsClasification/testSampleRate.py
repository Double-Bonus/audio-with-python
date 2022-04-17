# Basic Libraries
import pandas as pd
import numpy as np
import tensorflow as tf


# Libraries for Classification and building Models
from tensorflow import keras
from tensorflow.keras import callbacks

# Project Specific Libraries
import gc
import os
import librosa
import librosa.display
import torch
from datetime import datetime
import cv2
import sklearn
from sklearn.metrics import accuracy_score
from numpy import random

from cnn_model import get_cnn, get_simple_cnn_2
from visualise import draw_model_results



def load_spectograms():
    """ 
    Loads images to RAM from folder.

    Returns:
        Images arrau and class_name for y values
    """
    print("Loading images from drive to RAM!")
    img_data_array = np.zeros((DATA_SAMPLES_CNT, IMG_HEIGHT, IMG_WIDTH)) # some how it adds one pixcel
    class_name = np.zeros((DATA_SAMPLES_CNT, 1))
    
    cla = np.array(df["classID"]) # TODO FIX suppose its global

    for i in range(0, DATA_SAMPLES_CNT):
        image_path = "img_save//" + "out" + str(i+1) + "_" + str(df["class"][i]) + ".png"
        image= cv2.imread(image_path, cv2.COLOR_BGR2RGB) # TODO FIX: check color map
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



def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectrogram_image(y, sr, out_dir, out_name, hop_length, n_mels):
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
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    cv2.imwrite((out_dir + "\\" + out_name), img)


def save_wav_to_png(use_Kfold = False):
    """ 
    Saves spectograms data from sound files as png pictures
    """
    print("Saving pictures to drive")
    for i in range(DATA_SAMPLES_CNT):
        file_name = BASE_PATH  + "//audio//fold" + str(df["fold"][i]) + '//' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        y, sr = librosa.load(file_name, res_type='kaiser_fast', sr=22050*2) 
        
        img_name = 'out' + str(i+1) + "_" + str(df["class"][i]) + '.png'
        hop_length = 512           # number of samples per time-step in spectrogram
        n_mels = IMG_HEIGHT        # number of bins in spectrogram. Height of image
        time_steps = IMG_WIDTH - 1 # number of time-steps. Width of image (TODO FIX it add 1 px to width!!)
        
        

        y = librosa.util.utils.fix_length(y, sr * 4)  ## suppose it works????? It just adds white space!!!!
        # y, sr = pad_trunc(y, sr, 4000)
        
        start_sample = 0 # starting at beginning
        length_samples = time_steps * hop_length
        window = y[start_sample:start_sample+length_samples]
        
        if use_Kfold:
            dir_name = "processed//fold" + str(df["fold"][i])
        else:
            dir_name = "img_save"
        
        spectrogram_image(y=y, sr=sr, out_dir=dir_name , out_name=img_name, hop_length=hop_length, n_mels=n_mels)
    print("Done saving pictures!")
    
    
    
    
def train_CNN(X, Y, test_portion = 0.25):
    """ 
    Trains CNN with givrn inputs and predifend image dimensions

    Args:
        X : data inputs
        Y : data outputs
        test_portion (float, optional): What portion of data is used to validate. Defaults to 0.25.
    """
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=test_portion, random_state=7)
    
    x_train = x_train.reshape(x_train.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
    
    train_labels = keras.utils.to_categorical(y_train, num_classes=CLASSES_CNT)
    test_labels = keras.utils.to_categorical(y_test, num_classes=CLASSES_CNT)
    
    model = get_cnn(IMG_HEIGHT, IMG_WIDTH, CLASSES_CNT)
    model.summary()
    epochsCnt = 200
    earlystopper = callbacks.EarlyStopping(patience=epochsCnt*0.4, verbose=1, monitor='val_accuracy')
    checkpointer = callbacks.ModelCheckpoint('models\\urban_model.h5', verbose=1, save_best_only=True)
    
    hist = model.fit(x_train, train_labels, batch_size=64, epochs=epochsCnt, verbose=1, validation_data=(x_test, test_labels), callbacks = [earlystopper, checkpointer])
    draw_model_results(hist)
    
    
    
    
    
    
DEBUG_MODE = False
USE_KFOLD_VALID = False

BASE_PATH = "Urband_sounds//UrbanSound8K"
DATA_SAMPLES_CNT = 8732
CLASSES_CNT = 10
TEST_PORTION = 0.25
IMG_HEIGHT = 64
IMG_WIDTH = 345

start_time = datetime.now().strftime("%H:%M:%S")
tf.random.set_seed(0)
np.random.seed(0)

df = pd.read_csv("Urband_sounds//UrbanSound8K//metadata//UrbanSound8K.csv")

if USE_KFOLD_VALID:
    fold = "processed"
else:
    fold = "img_save"
    
# suppose existanse of images folder shows that there is data   NOW THIS APPPORACH IS RETARDED
if not os.path.exists(fold):
    save_wav_to_png()


X_data, Y_data = load_spectograms()
train_CNN(X_data, Y_data, TEST_PORTION)