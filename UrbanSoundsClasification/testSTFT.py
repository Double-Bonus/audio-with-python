y, sr = librosa.load(librosa.ex('trumpet'))

S = np.abs(librosa.stft(y))

S

S_left = librosa.stft(y, center=False)
D_short = librosa.stft(y, hop_length=64)


import matplotlib.pyplot as plt

fig, ax = plt.subplots()

img = librosa.display.specshow(librosa.amplitude_to_db(S,
                                                       ref=np.max),
                               y_axis='log', x_axis='time', ax=ax)

ax.set_title('Power spectrogram')

fig.colorbar(img, ax=ax, format="%+2.0f dB")