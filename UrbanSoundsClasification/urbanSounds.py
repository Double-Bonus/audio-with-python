# Basic implementation of UrbanSound8K dataset classification
# Using 75/25 train/test spilt achieved ~90-92% val. acc. 
# 10 folds cross validation ~72% acc.
# Based on: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

''' TODO:
   - Properly resize audio samples
   - Add Time Shift (optional?) https://towardsdatascience.com/audio-deep-learning-made-simple-part-3-data-preparation-and-augmentation-24c6e1f6b52
   - Update model structure to get better results
   - To get better result could try to combine multiple spectral features, for exemple Mel spectram, chromagram or tonnetz.
'''

# Basic Libraries
import pandas as pd
import numpy as np
import tensorflow as tf


# Libraries for Classification and building Models
from tensorflow import keras
from tensorflow.keras import callbacks

# Project Specific Libraries
import os
import librosa
import librosa.display
import torch
from datetime import datetime
import cv2
import sklearn
from numpy import random

# User defined modules
from visualise import draw_model_results, log_confusion_matrix
from cnn_model import *

# User defined classes
from datasetsBase import UrbandSound8k
from functionality import Functionality

#------------------ Normal work -----------------------

def train_kFold(use_chaged_speed):
    """
    Train and evaluate model via 10-Folds cross-validation

    Args:
        use_chaged_speed ([type=bool]): If changed speed audio is used
    """
    accuracies = np.zeros((CLASSES_CNT, 1))
    
    folds_cnt = np.zeros(CLASSES_CNT, dtype=int)
    for i in range(0, DATA_SAMPLES_CNT):
         folds_cnt[df["fold"][i] -1 ]  =  folds_cnt[df["fold"][i] -1] + 1
         
    urbandDb = UrbandSound8k(IMG_HEIGHT, IMG_WIDTH) # create class instance for dataset
           
    # 10-fold cross validation
    kfoldsCnt = 1
    for test_index in range(0, kfoldsCnt):

        
        model = get_cnn_minKernelReg(IMG_HEIGHT, IMG_WIDTH, CLASSES_CNT)
        if test_index == 0:
            model.summary() # show summary before first traing

        x_train, x_test, y_train, y_test = urbandDb.prepare_data_kFold(test_index, kfoldsCnt, folds_cnt)
        
        train_labels = keras.utils.to_categorical(y_train, num_classes=urbandDb.CLASSES_CNT)
        test_labels = keras.utils.to_categorical(y_test, num_classes=urbandDb.CLASSES_CNT)
        
        epochsCnt = 170
        earlystopper = callbacks.EarlyStopping(patience=epochsCnt*0.3, verbose=1, monitor='val_accuracy')
        checkpointer = callbacks.ModelCheckpoint('models\\k_urban_model.h5', verbose=1, monitor='val_accuracy', save_best_only=True)

        if 0: # use weight for class inbalandce
            clsWeight = urbandDb.get_class_weights()
            hist = model.fit(x_train, train_labels, epochs = epochsCnt, batch_size = 64, verbose = 1, class_weight = clsWeight,
               validation_data=(x_test, test_labels), callbacks = [earlystopper, checkpointer])
        else:
            hist = model.fit(x_train, train_labels, epochs = epochsCnt, batch_size = 64, verbose = 1,
               validation_data=(x_test, test_labels), callbacks = [earlystopper, checkpointer])
    
        model = keras.models.load_model('models\\k_urban_model.h5')
        pred = model.predict(x_test)
        y_pred = np.argmax(pred, axis=1) 

        rounded_labels=np.argmax(test_labels, axis=1) # from one hot to label, right?
        accuracies[test_index] = Functionality.calculate_accuracy(rounded_labels, y_pred)
        print("Temp k-Folds Accuracy: {0}".format(np.mean(accuracies)))

        Functionality.calculate_F1score(rounded_labels, y_pred)
        
        if 1: # initial results
            draw_model_results(hist)            
            log_confusion_matrix(model, x_test, y_test)
            
        if 1: # Save history for latte analysis
            np.save('my_history.npy',hist.history) # Load with history=np.load('my_history.npy',allow_pickle='TRUE').item() 
        

        
    print("\nAverage 10 Folds Accuracy: {0}".format(np.mean(accuracies)))
    print("Standart deviation of accuracy: {0}".format(np.std(accuracies)))

def pad_trunc(sig, sr, max_ms):
    """
    Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds

    Args:
        sig : audio signal data
    """
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

def spectrogram_image(y, sr, out_dir, out_name, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*4, hop_length=hop_length)
    # mels = librosa.feature.melspectrogram(y=y, sr=sr)
    
    if 1:
        mels = np.log(mels + 1e-9) # add small number to avoid log(0)
    else:  #testing !
        mels = np.mean(mels, axis=0)

    # min-max scale to fit inside 8-bit range
    img = Functionality.scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255 - img            # invert. make black==more energy

    # save as PNG
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    cv2.imwrite((out_dir + "\\" + out_name), img)
    
def save_wav_to_png(foldName, use_Kfold = False):
    """ 
    Saves spectograms data from sound files as png pictures
    """
    print("Saving pictures to drive")
    for i in range(DATA_SAMPLES_CNT):
        file_name = BASE_PATH  + "//audio//fold" + str(df["fold"][i]) + '//' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        y, sr = librosa.load(file_name, res_type='kaiser_fast') 
        
        img_name = 'out' + str(i+1) + "_" + str(df["class"][i]) + '.png'
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
        
        if use_Kfold:
            dir_name = "processed//fold" + str(df["fold"][i])
        else:
            dir_name = foldName
        
        spectrogram_image(y=window, sr=sr, out_dir=dir_name , out_name=img_name, hop_length=hop_length, n_mels=n_mels)
    print("Done saving pictures!")
    
def save_stretched_wav_to_png():
    """ 
    Saves spectograms data from sound files as png pictures but at random speed-up or slow-down sound
    """
    print("Saving stretched pictures to drive")
    for i in range(DATA_SAMPLES_CNT):
        file_name = BASE_PATH  + "//audio//fold" + str(df["fold"][i]) + '//' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        y, sr = librosa.load(file_name, res_type='kaiser_fast') 
        
        img_name = 'out' + str(i+1) + "_" + str(df["class"][i]) + '.png'
        hop_length = 512           # number of samples per time-step in spectrogram
        n_mels = IMG_HEIGHT        # number of bins in spectrogram. Height of image
        time_steps = IMG_WIDTH - 1 # number of time-steps. Width of image (TODO FIX it add 1 px to width!!)
        
        # changing speed!!!!
        y = librosa.effects.time_stretch(y, random.randint(4,17)*0.1)
        
        y = librosa.util.utils.fix_length(y, 88200)

        start_sample = 0 # starting at beginning
        length_samples = time_steps * hop_length
        # window = y[start_sample:start_sample+length_samples]

        dir_name = "speed//fold" + str(df["fold"][i])
        
        spectrogram_image(y=y, sr=sr, out_dir=dir_name , out_name=img_name, hop_length=hop_length, n_mels=n_mels)
    print("Done saving pictures!")
    
def load_spectograms(folder = "img_save"):
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
        image_path = folder + "//out" + str(i+1) + "_" + str(df["class"][i]) + ".png"
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
    
    model = get_cnn_minKernelReg_12(IMG_HEIGHT, IMG_WIDTH, CLASSES_CNT)
    model.summary()

    epochCnt = 160
    earlystopper = callbacks.EarlyStopping(patience=epochCnt*0.2, verbose=1, monitor='val_accuracy')
    checkpointer = callbacks.ModelCheckpoint('models\\urban_model.h5', verbose=1, monitor='val_accuracy', save_best_only=True)    
    
    hist = model.fit(x_train, train_labels, batch_size=128, epochs=epochCnt, verbose=1, validation_data=(x_test, test_labels), callbacks = [earlystopper, checkpointer])
    draw_model_results(hist)
    model = keras.models.load_model('models\\k_urban_model.h5')
    
    log_confusion_matrix(model, x_test, y_test)
    
    
# ----------------------- MAIN ------------------
DEBUG_MODE = False
USE_KFOLD_VALID = True

BASE_PATH = "Urband_sounds//UrbanSound8K"
DATA_SAMPLES_CNT = 8732
CLASSES_CNT = 10
TEST_PORTION = 0.25
IMG_HEIGHT = 128
IMG_WIDTH = 173 # 88.200 / hopsize (512) = 172.23

start_time = datetime.now().strftime("%H:%M:%S")
tf.random.set_seed(0)
np.random.seed(0)

df = pd.read_csv("Urband_sounds//UrbanSound8K//metadata//UrbanSound8K.csv")
if DEBUG_MODE:
    print(df.head())
    

if USE_KFOLD_VALID:
    fold = "processed"
else:
    fold = "img_save"
    # fold = "img_save_hop_612"
    # fold = "img_save04_11"
    
# suppose existanse of images folder shows that there is data   NOW THIS APPPORACH IS RETARDED
if not os.path.exists(fold):
    save_wav_to_png(fold, USE_KFOLD_VALID)
    
# if not os.path.exists("speed"):
#     save_stretched_wav_to_png()

if USE_KFOLD_VALID:
    train_kFold(use_chaged_speed=False)
else:
    X_data, Y_data = load_spectograms(fold)
    train_CNN(X_data, Y_data, TEST_PORTION)


print("Started at: ", start_time)
print("Done! at: ", datetime.now().strftime("%H:%M:%S"))