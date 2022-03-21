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

audio_fpath = "..//ESC-50-master//audio"
audio_clips = os.listdir(audio_fpath)
print("No. of .wav files in audio folder = ",len(audio_clips))

BASE_PATH = "..//ESC-50-master"
CSV_FILE_PATH = "..//ESC-50-master//meta//esc50.csv"  # path of csv file
SAVE_PATH = "img_esc"
samples_cnt = 2000
class_cnt = 50
imgH = 64
imgW = 128

TEST_10 = True

if 0:
    show_basic_data(BASE_PATH,useEsc50=True)



#reading the csv file
df = pd.read_csv(CSV_FILE_PATH)
print(df)





# Out of 50 classes we will be using 10 classes. dataframe has 
# column "esc10" which contains 10 classes. So, we will be using that 10 classes only.
# df_1 = df[df['fold'] == 1]

# print(df_1)

if TEST_10:
    df.drop(df.index[df['esc10'] == False], inplace = True)
    df = df.reset_index(drop=True)
    samples_cnt = 400
    class_cnt = 10
    
    classes = df['category'].unique()
    class_dict = {i:x for x,i in enumerate(classes)}
    df['target'] = df['category'].map(class_dict)


print("\n================================")
print(df.head(6))

print(samples_cnt)
print(class_cnt)


if not os.path.exists(SAVE_PATH):
    save_wav_to_png(df, DATA_SAMPLES_CNT = samples_cnt, BASE_PATH=BASE_PATH, IMG_HEIGHT = imgH, IMG_WIDTH = imgW)


X_data, Y_data = load_spectograms(df, DATA_SAMPLES_CNT = samples_cnt, IMG_HEIGHT = imgH, IMG_WIDTH = imgW)

train_CNN(X_data, Y_data, IMG_HEIGHT=imgH, IMG_WIDTH=imgW, CLASSES_CNT=class_cnt, simpleModel = False)