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
    def __init__(self, audioLen = (4*22050), urDb=UrbandSound8k()):
        self.urDb = urDb
        self.audioLen = audioLen


    # def train(x_train, y_train, x_test, y_test, num_classes, epochs):
    def train(self, xData, yData, num_classes=10, epochsCnt=40):

        if 1: # random spit
            x_train, x_test, y_train, y_first = sklearn.model_selection.train_test_split(xData, yData, test_size=0.20, random_state=7)
        else: # k-crosss valid
            x_train, x_test, y_train, y_first = self.urDb.prepare_data_kFold_LSTM_1dCNN(test_index = 0, kfoldsCnt = 10, X_data =xData , Y_data = yData)

        
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_first, num_classes)

        # # define the model
        # ann = Sequential()
        # ann.add(Dense(units=32, activation='relu', input_dim=x_train.shape[2]))

        # lstm = Sequential()
        # lstm.add(TimeDistributed(ann, input_shape=(x_train.shape[1], x_train.shape[2])))
        # lstm.add(LSTM(64, dropout=0.05, recurrent_dropout=0.05))
        # lstm.add(Dense(num_classes, activation='softmax'))

        # # print the model
        # f = open('Models\\model_annlstm.txt', 'w')
        # for i in range(0, len(ann.layers)):
        #     if i == 0:
        #         # print(cnn.summary())
        #         print(' ')
        #     print('{}. Layer {} with input / output shapes: {} / {}'.format(i, ann.layers[i].name, ann.layers[i].input_shape, ann.layers[i].output_shape))
        #     f.write('{}. Layer {} with input / output shapes: {} / {} \n'.format(i, ann.layers[i].name, ann.layers[i].input_shape, ann.layers[i].output_shape))
        #     if i == len(ann.layers) - 1:
        #         print(' ')
        # for i in range(0, len(lstm.layers)):
        #     if i == 0:
        #         # print(lstm.summary())
        #         # print(' ')
        #         f.write('\n')
        #     print('{}. Layer {} with input / output shapes: {} / {}'.format(i, lstm.layers[i].name, lstm.layers[i].input_shape, lstm.layers[i].output_shape))
        #     f.write('{}. Layer {} with input / output shapes: {} / {} \n'.format(i, lstm.layers[i].name, lstm.layers[i].input_shape, lstm.layers[i].output_shape))
        #     if i == len(lstm.layers) - 1:
        #         print(' ')
        # f.close()

        # # compile, fit, evaluate
        # lstm.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adamax(), metrics=['accuracy'])
        # lstm.fit(x_train, y_train, batch_size=512, epochs=epochs, verbose=2, validation_data=(x_test, y_test))
        # score = lstm.evaluate(x_test, y_test, batch_size=128)


        tempLenght = 1000 # really?
        audioLenght = 22050 * 4


        # create the model
        embedding_vecor_length = 8
        model = Sequential()
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
        else:
            model.add(Conv1D(128,
                        input_shape=[audioLenght, 1],
                        kernel_size=80,
                        strides=4,
                        padding='same',
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(l=0.0001)))
            # model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size=4, strides=None))

            model.add(Conv1D(128,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(l=0.0001)))
            # model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size=4, strides=None))
            model.add(Dropout(0.1))
            
            model.add(Conv1D(256,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(l=0.0001)))
            # model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size=4, strides=None))
            model.add(Dropout(0.25))

            model.add(Conv1D(512,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(l=0.0001)))
            # model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling1D(pool_size=4, strides=None))
            model.add(Dropout(0.25))

            model.add(LSTM(units=8)) 

            # model.add(Lambda(lambda x: K.mean(x, axis=1)))  # Same as GAP for 1D Conv Layer

        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        hist = model.fit(x_train, y_train, epochs=epochsCnt, batch_size=32, validation_data=(x_test, y_test))

        if 1: # initial results
            draw_model_results(hist)            
            log_confusion_matrix(model, x_test, y_first)
            
        if 1: # Save history for latter analysis
            np.save('my_history.npy',hist.history) # Load with history=np.load('my_history.npy',allow_pickle='TRUE').item() 
        




    def load_audio(self):
        print("Loading audio files from drive!")

        cnt = 300
        # cnt = self.urDb.DATA_SAMPLES_CNT
    
        audioArray = np.zeros((cnt, self.audioLen), dtype=np.uint8)
        className = np.zeros((cnt, 1))
        
        for i in range(cnt):
            file_name = self.urDb.BASE_PATH  + "//audio//fold" + str(self.urDb.df["fold"][i]) + '//' + self.urDb.df["slice_file_name"][i]

            y, sr = librosa.load(file_name, res_type='kaiser_fast') 
            y = librosa.util.utils.fix_length(y, self.audioLen)

            # y_8k = librosa.resample(y, orig_sr=sr, target_sr=8000)
            # audio8 = Functionality.scale_minmax(y_8k, 0, 255).astype(np.uint8)
            audio8 = Functionality.scale_minmax(y, 0, 255).astype(np.uint8)

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

    instance = lstmRaw()


    xData, yData = instance.load_audio()
    # train_CNN(X_data, Y_data, TEST_PORTION)
    # print(xData)
    # print(yData)

    print(xData.shape)



    instance.train(xData, yData)


    print("Started at: ", start_time)
    print("Done! at: ", datetime.now().strftime("%H:%M:%S"))
    print("End from LSTM ANN!")


if __name__ == "__main__":
    main()
   