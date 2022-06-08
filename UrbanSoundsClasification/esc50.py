# Basic Libraries
import pandas as pd
import numpy as np
import tensorflow as tf


import os
from time import sleep
import matplotlib.pyplot as plt

#for loading and visualizing audio files
import librosa
import librosa.display

# User defined modules
from visualise import *
from functionality import *

from datasetsBase import ESC10



def getEscTestData(X, Y, testIdx, dataset):
    # folds are spillt in 400 in the db names order 
    
    x_train = np.zeros((1600, dataset.IMG_HEIGHT, dataset.IMG_WIDTH))
    x_test =  np.zeros((400,  dataset.IMG_HEIGHT, dataset.IMG_WIDTH))
    y_train = np.zeros((1600, 1))
    y_test =  np.zeros((400, 1))
    
    i_test = 0
    i_train = 0

    for i in range(dataset.samplesCnt):
        if dataset.df['fold'][i] == testIdx:
            y_test[i_test] = Y[i]
            x_test[i_test] = X[i]
            i_test = i_test + 1
        else:
            y_train[i_train] = Y[i]
            x_train[i_train] = X[i]  
            i_train = i_train + 1

    return x_train, x_test, y_train, y_test


def train_CNN(X, Y, db, test_portion = 0.20):
    """ 
    Trains CNN with givrn inputs and predifend image dimensions

    Args:
        X : data inputs
        Y : data outputs
        test_portion (float, optional): What portion of data is used to validate. Defaults to 0.25.
    """

    for iTest in range(1, 2):
        if 1: # random spilt 
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=test_portion, random_state=7)
        else:
            x_train, x_test, y_train, y_test = getEscTestData(X, Y, testIdx=iTest, dataset=db)

        print("Using fold nr: " + str (iTest))
        x_train = x_train.reshape(x_train.shape[0], db.IMG_HEIGHT, db.IMG_WIDTH, 1)
        x_test = x_test.reshape(x_test.shape[0], db.IMG_HEIGHT, db.IMG_WIDTH, 1)
        
        train_labels = keras.utils.to_categorical(y_train, num_classes=db.classesCnt)
        test_labels = keras.utils.to_categorical(y_test, num_classes=db.classesCnt)
        
        model = get_cnn_minKernel_smallerL2_2(db.IMG_HEIGHT, db.IMG_WIDTH, db.classesCnt)
        model.summary()

        epochCnt = 400
        epochCnt = 500
        earlystopper = callbacks.EarlyStopping(patience=epochCnt*0.20, verbose=1, monitor='val_accuracy')
        checkpointer = callbacks.ModelCheckpoint('models\\esc50.h5', verbose=1, monitor='val_accuracy', save_best_only=True)
            
        hist = model.fit(x_train, train_labels, batch_size=64, epochs=epochCnt, verbose=1, 
            validation_data=(x_test, test_labels), callbacks = [earlystopper, checkpointer])

        draw_model_results(hist.history, saveFig=True)
        model = keras.models.load_model('models\\esc50.h5')
        log_confusion_matrix(model, x_test, y_test)



if __name__ == "__main__":
    print("Training for esc-50")

    TEST_10 = False
    SAVE_PATH = "img_esc"

    visHelper = Visualise(urDb=ESC50())

    if 0:
        visHelper.show_basic_data(useEsc50=True)

    print(visHelper.urDb.df)

    esc10_db = ESC50()
    print("\n================================")
    print(esc10_db.df.head(6))

    print(esc10_db.samplesCnt)
    print(esc10_db.classesCnt)


    if not os.path.exists(SAVE_PATH):
        # Functionality.save_wav_to_png(esc10_db.df, DATA_SAMPLES_CNT = esc10_db.df.samplesCnt, 
        #     BASE_PATH=esc10_db.df.BASE_PATH, IMG_HEIGHT = esc10_db.df.height, IMG_WIDTH = esc10_db.df.width) PADARYT TAIP
        save_wav_to_png(esc10_db.df, DATA_SAMPLES_CNT = esc10_db.samplesCnt, 
            BASE_PATH=esc10_db.BASE_PATH, IMG_HEIGHT = esc10_db.IMG_HEIGHT, IMG_WIDTH = esc10_db.IMG_WIDTH)


    X_data, Y_data = load_spectograms(esc10_db.df, DATA_SAMPLES_CNT = esc10_db.samplesCnt,
    IMG_HEIGHT = esc10_db.IMG_HEIGHT, IMG_WIDTH = esc10_db.IMG_WIDTH)


    imgH = esc10_db.IMG_HEIGHT
    imgW = esc10_db.IMG_WIDTH
    train_CNN(X_data, Y_data, db = esc10_db)



