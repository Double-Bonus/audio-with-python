
#do not auto import!
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy  # Python library used for scientific computing and technical computing
import sys
import pickle
import librosa
import librosa.display
from IPython.display import Audio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
import tensorflow as tf
from tensorflow import keras
#do not auto import!


print("Labas!")

# df = pd.read_csv("/Data/features_30_sec.csv")
# FIX THIS
df = pd.read_csv("D:/Backup/Desktop/I/KTU/7Semestras/Intelektika/Projektas/music_genre_classf/Data/features_3_sec.csv")


# pd.read_csv("../data_folder/data.csv")
print(df.head()) # gives first n lines of data 

print(df.shape)
df = df.drop(labels='filename', axis=1)



audio_recording = ("D:/Backup/Desktop/I/KTU/7Semestras/Intelektika/Projektas/music_genre_classf/Data/genres_original/country/country.00050.wav")
data, sr = librosa.load(audio_recording)
print(type(data), type(sr))

librosa.load(audio_recording, sr=45600)


# plot as Waveforms 
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, color = "blue")
plt.show()


# Spectrograms
stft = librosa.stft(data)
stft_db = librosa.amplitude_to_db(abs(stft))
plt.figure(figsize=(14,6))
librosa.display.specshow(stft, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.show()

# white spectro
librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
plt.show()


# Spectral Rolloff
spectral_rollof = librosa.feature.spectral_rolloff(data+0.01, sr=sr)[0]
plt.figure(figsize=(12,4))
librosa.display.waveshow(data, sr=sr, alpha=0.4, color='grey')
plt.show()

# Chroma Feature
chroma = librosa.feature.chroma_stft(data, sr=sr)
plt.figure(figsize=(16,6))
librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', cmap='coolwarm')
plt.colorbar()
plt.title('Croma Features')
plt.show()


# Zero Crossing Rate
plt.figure(figsize=(14,5))
plt.plot(data[1000:1200])
plt.grid()
plt.show()
zero_cross_rate = librosa.zero_crossings(data[1000:1200], pad = False)
print('Number of zero-crossings is :', sum(zero_cross_rate))




