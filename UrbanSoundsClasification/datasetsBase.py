import pandas as pd
import numpy as np
import cv2
from tensorflow import keras


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
        
        train_labels = keras.utils.to_categorical(y_train, num_classes=self.CLASSES_CNT)
        test_labels = keras.utils.to_categorical(y_test, num_classes=self.CLASSES_CNT)

        return x_train, x_test, train_labels, test_labels


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




