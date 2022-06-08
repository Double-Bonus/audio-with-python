# Basic Libraries
from pyexpat import model
import pandas as pd
import numpy as np
import tensorflow as tf


# Libraries for Classification and building Models
from tensorflow import keras
from tensorflow.keras import callbacks


# Project Specific Libraries
import os, sys
import librosa
import librosa.display
import torch
from datetime import datetime
import cv2
import sklearn
from numpy import random

# User defined modules
# from visualise import draw_model_results, log_confusion_matrix


# User defined classes
from datasetsBase import UrbandSound8k
from functionality import Functionality

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Conv1D, MaxPooling1D 



from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import LeakyReLU, ELU
from tensorflow.keras import regularizers
from keras.layers import TimeDistributed, LSTM, ConvLSTM2D 
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dense

from keras.layers import Lambda
import keras.backend as K

from visualise import draw_model_results, log_confusion_matrix


class lstmRaw:
    def __init__(self, sr = 22050, urDb=UrbandSound8k()):
        self.urDb = urDb
        self.samplingRate = sr
        self.audioLen = sr * 4

    def getModel(self, x_train):
        model = Sequential()

        tempLenght = 1000 # really?
        audioLenght = 22050 * 4
        # create the model
        embedding_vecor_length = 8


        if 0: # low val acc
            model.add(Embedding(tempLenght, embedding_vecor_length, input_length=audioLenght)) # e.g. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
            model.add(LSTM(32))
        elif 0:
            model.add(LSTM(8, input_shape=(audioLenght, 1))) #doesnt work on its own
        elif 0:
            # model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.20, return_sequences=True,input_shape = (audioLenght,1)))
            # model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.20, return_sequences=False))
            model.add(LSTM(units=16, dropout=0.05, return_sequences=True, input_shape = (audioLenght,1)))
            model.add(LSTM(units=8, dropout=0.05, return_sequences=False))
            model.add(Dense(20,activation='relu')) 
        elif 0:
            cnn = Sequential()
            
            cnn.add(Conv1D(128, input_shape=(audioLenght, 1),
                kernel_size=80,
                strides=4, padding='same',
                kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(l=0.0001)))
            cnn.add(Activation('relu'))
            cnn.add(MaxPooling1D(pool_size=4, strides=None))

            cnn.add(Conv1D(128, kernel_size=3, strides=1,padding='same', kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(l=0.0001)))
            cnn.add(Activation('relu'))
            cnn.add(MaxPooling1D(pool_size=4, strides=None))
            cnn.add(Dropout(0.1))
            
            cnn.add(Conv1D(256, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(l=0.0001)))
            cnn.add(Activation('relu'))
            cnn.add(MaxPooling1D(pool_size=4, strides=None))
            cnn.add(Dropout(0.25))

            cnn.add(Conv1D(512, kernel_size=3, strides=1, padding='same',
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(l=0.0001)))
            cnn.add(Activation('relu'))
            cnn.add(MaxPooling1D(pool_size=4, strides=None))
            cnn.add(Dropout(0.25))
            cnn.add(Flatten())
            # model.add(TimeDistributed(cnn, input_shape=(None, audioLenght, 1) ))
            
            # model.add(LSTM(6, input_shape=(None,None, audioLenght, 1)))


            model.add(TimeDistributed(cnn, input_shape=(audioLenght,1)))
            model.add(LSTM(6, dropout=0.0, recurrent_dropout=0.0)) # makes trainig way longer
            # , input_shape = (None, 8, 86, 512)
            # model.add(Dense(...))

            # model.add(Lambda(lambda x: K.mean(x, axis=1)))  # Same as GAP for 1D Conv Layer

        else:
            print(x_train.shape)
            print("-------------------------")
            cnn = Sequential()
            cnn.add(Conv1D(128, input_shape=(x_train.shape[2],),
                kernel_size=80,
                strides=4, padding='same',
                kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(l=0.0001)))
            cnn.add(Activation('relu'))
            cnn.add(MaxPooling1D(pool_size=4, strides=None))

            cnn.add(Conv1D(128, kernel_size=3, strides=1,padding='same', kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(l=0.0001)))
            cnn.add(Activation('relu'))
            cnn.add(MaxPooling1D(pool_size=4, strides=None))
            cnn.add(Dropout(0.1))
            
            cnn.add(Conv1D(256, kernel_size=3, strides=1, padding='same', kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(l=0.0001)))
            cnn.add(Activation('relu'))
            cnn.add(MaxPooling1D(pool_size=4, strides=None))
            cnn.add(Dropout(0.25))

            cnn.add(Conv1D(512, kernel_size=3, strides=1, padding='same',
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(l=0.0001)))
            cnn.add(Activation('relu'))
            cnn.add(MaxPooling1D(pool_size=4, strides=None))
            cnn.add(Dropout(0.25))
            cnn.add(Flatten())
            cnn.add(Dense(100))
            cnn.summary()

            # model.add(TimeDistributed(cnn, input_shape=(None, audioLenght, 1) ))
            
            # model.add(LSTM(6, input_shape=(None,None, audioLenght, 1)))


            model.add(TimeDistributed(cnn, input_shape=(4, 22050, 1) ))
            model.add(LSTM(6, dropout=0.0, recurrent_dropout=0.0)) # makes trainig way longer
            # , input_shape = (None, 8, 86, 512)
            # model.add(Dense(...))

            # model.add(Lambda(lambda x: K.mean(x, axis=1)))  # Same as GAP for 1D Conv Layer

        model.add(Dense(self.urDb.CLASSES_CNT, activation='softmax'))
        model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        return model

    def getModel_1dCNN(self):
        model = Sequential()

        model.add(Conv1D(48,
                input_shape=[self.audioLen, 1],
                kernel_size=80, strides=4, padding='valid',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(l=0.0001)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4, strides=None))

        model.add(Conv1D(72, kernel_size=3, strides=1, padding='valid',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(l=0.0001)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4, strides=None))
        model.add(Dropout(0.1))
            
        model.add(Conv1D(108, kernel_size=3, strides=1, padding='valid',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(l=0.0001)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4, strides=None))
        model.add(Dropout(0.25))

        model.add(Conv1D(172, kernel_size=3, strides=1, padding='valid',
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(l=0.0001)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4, strides=None))
        model.add(Dropout(0.25))
        model.add(Lambda(lambda x: K.mean(x, axis=1)))  # Same as GAP for 1D Conv Layer
        
        model.add(Dense(self.urDb.CLASSES_CNT, activation='softmax'))
        print(model.summary())
        return model

    def getModel_1dCNN_smaller(self):
        model = Sequential()

        model.add(Conv1D(48,
                input_shape=[self.audioLen, 1],
                kernel_size=80, strides=4, padding='valid',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(l=0.0001)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4, strides=None))

        model.add(Conv1D(72, kernel_size=3, strides=1, padding='valid',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(l=0.0001)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4, strides=None))
        model.add(Dropout(0.25))
            
        model.add(Conv1D(108, kernel_size=3, strides=1, padding='valid',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(l=0.0001)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4, strides=None))
        model.add(Dropout(0.25))

        model.add(Lambda(lambda x: K.mean(x, axis=1)))  # Same as GAP for 1D Conv Layer
        
        model.add(Dense(self.urDb.CLASSES_CNT, activation='softmax'))
        print(model.summary())
        return model

    def getModel_1dCnnLstm(self, X_train):
        #Build the model
        print(X_train.shape)


        # sr = 8000
        # n_steps, n_length = 20, 1600
        n_steps, n_length = 4, 8000
        
        
        
        X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, 1))
        # define model
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=32, kernel_size=9, activation='relu'), input_shape=(None,n_length,1)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=8)))

        model.add(TimeDistributed(Conv1D(filters=48, kernel_size=5, activation='relu')))
        model.add(TimeDistributed(Dropout(0.1)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=8)))

        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
        model.add(TimeDistributed(Dropout(0.25)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=8)))

        # model.add(TimeDistributed(Flatten()))

        model.add(keras.layers.LSTM(16, return_sequences=True))
        model.add(keras.layers.LSTM(16))
        # dense layer
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dropout(0.25))
        model.add(Flatten())
        
        model.add(Dense(64))


        model.add(Dense(10, activation='softmax'))
        print(model.summary())

        return model




    def train(self, xData, yData, num_classes=10, epochsCnt=160):

        # argVal = int(sys.argv[1]) 
        argVal = 0

        if 0: # random spit
            x_train, x_test, y_train, y_first = sklearn.model_selection.train_test_split(xData, yData, test_size=0.20, random_state=7)
        else: # k-crosss valid
            x_train, x_test, y_train, y_first = self.urDb.prepare_data_kFold_LSTM_1dCNN(test_index = argVal, kfoldsCnt = 10, X_data =xData , Y_data = yData, 
            audioLen=self.audioLen)

        
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_first, num_classes)

        if 0:
            model = self.getModel(x_train)
        elif 0:
            model = self.getModel_1dCNN()
            # model = self.getModel_1dCNN_smaller()
        else: 
            model = self.getModel_1dCnnLstm(x_train)
            # sr = 8000

            # n_steps, n_length = 20, 1600
            n_steps, n_length = 4, 8000
            x_train = x_train.reshape((x_train.shape[0], n_steps, n_length, 1))
            x_test =  x_test.reshape((x_test.shape[0], n_steps, n_length, 1))

            



        earlystopper = callbacks.EarlyStopping(patience=epochsCnt*0.25, verbose=1, monitor='val_accuracy')
        checkpointer = callbacks.ModelCheckpoint('models\\rawDataModel.h5', verbose=1, monitor='val_accuracy', save_best_only=True)
        model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

        hist = model.fit(x_train, y_train, epochs=epochsCnt, batch_size=32, validation_data=(x_test, y_test),
         callbacks=[earlystopper, checkpointer], verbose = 1)


        if 0: # initial results
            draw_model_results(hist)            
            log_confusion_matrix(model, x_test, y_first)
            
        # if 1: # Save history for latter analysis
        #     np.save('my_history.npy',hist.history) # Load with history=np.load('my_history.npy',allow_pickle='TRUE').item() 


        model = keras.models.load_model('models\\rawDataModel.h5')
        pred = model.predict(x_test)
        y_pred = np.argmax(pred, axis=1) 
        rounded_labels=np.argmax(y_test, axis=1)
        
        av_macro, av_weighted = Functionality.calculate_F1score(rounded_labels, y_pred)
        acc = Functionality.calculate_accuracy(rounded_labels, y_pred)
        
        print('----------------------------')
        print('Accuracy: ', acc)
        print('Macro F1-score: ', av_macro)
        print('Weighted F1-score: ', av_weighted)
        
    def load_audio(self):
        print("Loading audio files from drive!")

        # cnt = 1000
        cnt = self.urDb.DATA_SAMPLES_CNT
    
        audioArray = np.zeros((cnt, self.audioLen), dtype=np.uint8)
        className = np.zeros((cnt, 1))
        
        for i in range(cnt):
            file_name = self.urDb.BASE_PATH  + "//audio//fold" + str(self.urDb.df["fold"][i]) + '//' + self.urDb.df["slice_file_name"][i]

            y, sr = librosa.load(file_name, res_type='kaiser_fast') 
            if (sr != self.samplingRate):
                y = librosa.resample(y, orig_sr=sr, target_sr=self.samplingRate)
            y = librosa.util.utils.fix_length(y, self.audioLen)
            audio8 = Functionality.scale_minmax(y, 0, 255).astype(np.uint8)
            # audio8 = Functionality.scale_minmax(y, 0, 255).astype(np.uint8)

            audioArray[i] = audio8
            className[i] = self.urDb.df["classID"][i]
            
            
        print("Done loading audio!")
        return audioArray, className



def main():

    print("Started LSTM ANN")
    
    start_time = datetime.now().strftime("%H:%M:%S")
    tf.random.set_seed(0)   
    np.random.seed(0)

    # urbandDb = UrbandSound8k() # create default class instance for dataset

    instance = lstmRaw(sr=8000)
    xData, yData = instance.load_audio()
    # train_CNN(X_data, Y_data, TEST_PORTION)
    # print(xData)
    # print(yData)

    # print(xData.shape)

    # xData = np.reshape(xData, (1000, 4, 22050))

    # print(xData.shape)
    


    instance.train(xData, yData)


    print("Started at: ", start_time)
    print("Done! at: ", datetime.now().strftime("%H:%M:%S"))
    print("End from LSTM ANN!")


if __name__ == "__main__":
    main()
   