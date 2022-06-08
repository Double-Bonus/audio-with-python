import pandas as pd
import numpy as np
import cv2
from tensorflow import keras

# For visualization
import librosa
import matplotlib.pyplot as plt


# TODO move here all functions that uses df !!!!

class UrbandSound8k:
    def __init__(self, height = 128, width = 128):
        self.IMG_HEIGHT = height
        self.IMG_WIDTH = width
        self.BASE_PATH = "Urband_sounds//UrbanSound8K"
        self.DATA_SAMPLES_CNT = 8732
        self.CLASSES_CNT = 10
        self.df = pd.read_csv(self.BASE_PATH + "//metadata//UrbanSound8K.csv")
        self.FRAME_CNT = 1
        self.foldsCnt = 10
        
    def get_class_weights(self):
        """ 
        Gets class weights for urbanSound8k dataset,

        Should be used in model.fit() as class_weight = clsWeight
        """

        # all have 1k records except,  car_horn(429), gun_shot(374), siren (929)
        cl_cnt = [1000, 429, 1000, 1000, 1000, 1000, 374, 1000, 929, 1000]
        weights = [None] * self.CLASSES_CNT
        for i in range(self.CLASSES_CNT):
            weights[i] = (1 / cl_cnt[i]) * (self.DATA_SAMPLES_CNT / self.CLASSES_CNT)

        class_weight = {0: weights[0],
        1: weights[1],
        2: weights[2],
        3: weights[3],
        4: weights[4],
        5: weights[5],
        6: weights[6],
        7: weights[7],
        8: weights[8],
        9: weights[9]
        }
        return class_weight

    # TODO: this could be calcualted before traing
    def calculate_class_imbalance(self, testIndex):
        cl_cnt = [1000, 429, 1000, 1000, 1000, 1000, 374, 1000, 929, 1000]
        
        testCnt = [0] * self.CLASSES_CNT

        # caclulate number of samples in fold and subtract for total to get number of samples in tests (faster than calculating all train)
        for i in range(0, self.DATA_SAMPLES_CNT):
            # print("aaaa")
            if(testIndex == (self.df["fold"][i] - 1)):
                testCnt[self.df["classID"][i]] = testCnt[self.df["classID"][i]] + 1
        print(testCnt)
        
        trainCnt = [0] * self.CLASSES_CNT
        for i in range(0, self.CLASSES_CNT):
            trainCnt[i] = cl_cnt[i] - testCnt[i]

        print(trainCnt)

        weights = [0] * self.CLASSES_CNT
        for i in range(self.CLASSES_CNT):
            weights[i] = (1 / trainCnt[i]) * (np.sum(trainCnt) / self.CLASSES_CNT)
        
        class_weight = {0: weights[0],
        1: weights[1],
        2: weights[2],
        3: weights[3],
        4: weights[4],
        5: weights[5],
        6: weights[6],
        7: weights[7],
        8: weights[8],
        9: weights[9]
        }
        return class_weight

    def prepare_data_kFold(self, test_index, kfoldsCnt, folds_cnt, use_chaged_speed = False):
        print("Using " + str(test_index+1) + " fold out of: " + str(kfoldsCnt))
        x_train = np.zeros((self.DATA_SAMPLES_CNT - folds_cnt[test_index], self.IMG_HEIGHT, self.IMG_WIDTH))
        y_train = np.zeros((self.DATA_SAMPLES_CNT - folds_cnt[test_index], 1))
        x_test = np.zeros((folds_cnt[test_index], self.IMG_HEIGHT, self.IMG_WIDTH))
        y_test = np.zeros((folds_cnt[test_index], 1))
        test_i = 0
        train_i = 0
        
        for i in range(0, self.DATA_SAMPLES_CNT):
            if (test_index != (self.df["fold"][i]-1)) and (use_chaged_speed):
                image_path = "speed//fold" + str(self.df["fold"][i]) + "//out" + str(i+1) + "_" + str(self.df["class"][i]) + ".png"
            else:
                image_path = "processed//fold" + str(self.df["fold"][i]) + "//out" + str(i+1) + "_" + str(self.df["class"][i]) + ".png"
            image= cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            if image is None:
                print("Error, image was not found from: " + image_path)
                quit()
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            if test_index == (self.df["fold"][i]-1):
                x_test[test_i] = image 
                y_test[test_i] = self.df["classID"][i]
                test_i = test_i + 1
            else:
                x_train[train_i] = image 
                y_train[train_i] = self.df["classID"][i]
                train_i = train_i + 1
                    
        x_train = x_train.reshape(x_train.shape[0], self.IMG_HEIGHT, self.IMG_WIDTH, 1)
        x_test = x_test.reshape(x_test.shape[0], self.IMG_HEIGHT, self.IMG_WIDTH, 1)
        
        return x_train, x_test, y_train, y_test


    # LSTM -------------------------------------------------------------------
    def set_frame_cnt(self, frameCnt):
        self.FRAME_CNT = frameCnt

    def prepare_data_kFold_LSTM(self, test_index, kfoldsCnt, folds_cnt, X_data, Y_data):
        print("Using " + str(test_index+1) + " fold out of: " + str(kfoldsCnt))
        x_train = np.zeros((self.DATA_SAMPLES_CNT - folds_cnt[test_index], self.FRAME_CNT, self.IMG_HEIGHT, (self.IMG_WIDTH // self.FRAME_CNT)))
        x_test =  np.zeros((folds_cnt[test_index],                         self.FRAME_CNT, self.IMG_HEIGHT, (self.IMG_WIDTH // self.FRAME_CNT)))
        y_train = np.zeros((self.DATA_SAMPLES_CNT - folds_cnt[test_index], 1))
        y_test =  np.zeros((folds_cnt[test_index], 1))

        test_i = 0
        train_i = 0
        for i in range(0, self.DATA_SAMPLES_CNT):
            if test_index == (self.df["fold"][i]-1):
                x_test[test_i] = X_data[i] 
                y_test[test_i] = Y_data[i]
                test_i = test_i + 1
            else:
                x_train[train_i] = X_data[i] 
                y_train[train_i] = Y_data[i]
                train_i = train_i + 1

        x_train = x_train.reshape(x_train.shape[0],  self.FRAME_CNT, self.IMG_HEIGHT, (self.IMG_WIDTH // self.FRAME_CNT), 1)
        x_test =  x_test.reshape( x_test.shape[0],   self.FRAME_CNT, self.IMG_HEIGHT, (self.IMG_WIDTH // self.FRAME_CNT), 1)

        return x_train, x_test, y_train, y_test


    # LSTM raw -------------------------------------------------------------------
    # def HARCODINTI???
    def calculate_number_of_classes(self):
        folds_cnt = np.zeros(self.CLASSES_CNT, dtype=int)
        for i in range(0, self.DATA_SAMPLES_CNT):
            folds_cnt[self.df["fold"][i] -1 ]  =  folds_cnt[self.df["fold"][i] -1] + 1
        return folds_cnt

    # fold count turetu buti klases viduje!
    def prepare_data_kFold_LSTM_1dCNN(self, test_index, kfoldsCnt, X_data, Y_data, audioLen = (4*22050)):
        folds_cnt = self.calculate_number_of_classes()
        print("Using " + str(test_index+1) + " fold out of: " + str(kfoldsCnt))
        x_train = np.zeros((self.DATA_SAMPLES_CNT - folds_cnt[test_index], audioLen))
        x_test =  np.zeros((folds_cnt[test_index],                         audioLen))
        y_train = np.zeros((self.DATA_SAMPLES_CNT - folds_cnt[test_index], 1))
        y_test =  np.zeros((folds_cnt[test_index], 1))

        test_i = 0
        train_i = 0
        for i in range(0, self.DATA_SAMPLES_CNT):
            if test_index == (self.df["fold"][i]-1):
                x_test[test_i] = X_data[i] 
                y_test[test_i] = Y_data[i]
                test_i = test_i + 1
            else:
                x_train[train_i] = X_data[i] 
                y_train[train_i] = Y_data[i]
                train_i = train_i + 1

        # x_train = x_train.reshape(x_train.shape[0],  self.FRAME_CNT, self.IMG_HEIGHT, (self.IMG_WIDTH // self.FRAME_CNT), 1)
        # x_test =  x_test.reshape( x_test.shape[0],   self.FRAME_CNT, self.IMG_HEIGHT, (self.IMG_WIDTH // self.FRAME_CNT), 1)

        return x_train, x_test, y_train, y_test


class ESC50:
    def __init__(self, height = 128, width = 173):
        self.IMG_HEIGHT = height
        self.IMG_WIDTH = width
        self.BASE_PATH = "..//ESC-50"
        # self.samplesCnt = 2000
        self.samplesCnt = 2060
        self.classesCnt = 50
        self.df = pd.read_csv(self.BASE_PATH + "//meta//esc50.csv")

class ESC10:
    def __init__(self, height = 128, width = 216):
        self.IMG_HEIGHT = height
        self.IMG_WIDTH = width
        self.BASE_PATH = "..//ESC-50"
        self.samplesCnt = 400
        self.classesCnt = 10

        self.df = pd.read_csv(self.BASE_PATH + "//meta//esc50.csv")
        self.df.drop(self.df.index[self.df['esc10'] == False], inplace = True)
        self.df = self.df.reset_index(drop=True)

        # rename 'target' collum from 0-49 -> 0-9
        self.df['target'][self.df['target'] == 10] = 2
        self.df['target'][self.df['target'] == 11] = 3
        self.df['target'][self.df['target'] == 12] = 4
        self.df['target'][self.df['target'] == 20] = 5
        self.df['target'][self.df['target'] == 21] = 6
        self.df['target'][self.df['target'] == 38] = 7
        self.df['target'][self.df['target'] == 40] = 8
        self.df['target'][self.df['target'] == 41] = 9

"""""        
0	dog	TRUE
1	rooster	TRUE
10	rain	TRUE
11	sea_waves	TRUE
12	crackling_fire	TRUE

20	crying_baby	TRUE
21	sneezing	TRUE
38	clock_tick	TRUE
40	helicopter	TRUE
41	chainsaw	TRUE
"""""        
