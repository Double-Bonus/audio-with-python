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
from sklearn.preprocessing import LabelEncoder, normalize, StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Dropout



def draw_data():
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

def train_model(model, optimizer, epochs = 32):
    batch_size = 128
    model.compile(optimizer=optimizer,
                  loss = 'sparse_categorical_crossentropy',
                  metrics = 'accuracy'
    )
    return model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
                    batch_size = batch_size)
    

def plot_validation(history):
    print("Validation acc ", max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12,6))
    plt.show()

df = pd.read_csv('Data\\features_3_sec.csv')

print(df.head()) # gives first n lines of data 

print(df.shape)
df = df.drop(labels='filename', axis=1)


audio_recording = ('Data\\genres_original\\country\\country.00050.wav')

data, sr = librosa.load(audio_recording)
print(type(data), type(sr))

librosa.load(audio_recording, sr=45600)

# draws our audio file in differerent formats
#draw_data()

class_list = df.iloc[:,-1]
convertor = LabelEncoder()

y = convertor.fit_transform(class_list)
print(y)

fit = StandardScaler()
X = fit.fit_transform(np.array(df.iloc[:,:-1], dtype=float)) # kodel float ne np.float????

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10, activation='softmax'))

print(model.summary())
model_history = train_model(model=model, epochs=200, optimizer='adam')

test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=128)
print("The test los is :",  test_loss)
print("\nThe best acc is: ", test_acc*100)

plot_validation(model_history)