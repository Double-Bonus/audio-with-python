# File for handling plots and visualising data

import sklearn
import matplotlib.pyplot as plt
import numpy as np
import itertools
import librosa
import librosa.display


from datasetsBase import UrbandSound8k

# ----------------------------- Private Defines ---------------------------------


# ----------------------------- Private functions -------------------------------
def _plot_confusion_matrix(cm, class_names):
    """ 
    Saves a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)
                       [:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    # TODO why need itertools???
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('ConfusionMatrix.png')


# ----------------------------- Public functions --------------------------------
def log_confusion_matrix(model, test_images, test_labels):
    # Use the model to predict the values from the validation dataset.
    test_pred_raw = model.predict(test_images)
    test_pred = np.argmax(test_pred_raw, axis=1)

    print("Saving results in confusion matrix")

    # Hardcoded for now TODO FIX
    class_names = [
        'air_conditioner',
        'car_horn',
        'children_playing',
        'dog_bark',
        'drilling',
        'engine_idling',
        'gun_shot',
        'jackhammer',
        'siren',
        'street_music']

    # Calculate the confusion matrix.
    cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
    # Log the confusion matrix as an image summary.
    _plot_confusion_matrix(cm, class_names=class_names)


def draw_model_results(model_history):
    plt.subplot(121)
    plt.plot(model_history.history['accuracy'], 'r')
    plt.plot(model_history.history['val_accuracy'], 'b')
    plt.ylabel('Tikslumas, r - train, b - val')
    plt.xlabel('Epocha')
    plt.grid(b=True)

    plt.subplot(122)
    plt.plot(model_history.history['loss'], 'r')
    plt.plot(model_history.history['val_loss'], 'b')
    plt.ylabel('Nuostolis (angl. loss), r - train, b - val')
    plt.xlabel('Epocha')
    plt.grid(b=True)
    plt.show()


# ------------------------ CLASS ----------------------
class Visualise:
    def __init__(self, urDb=UrbandSound8k()):
        self._WINDOW_HEIGHT = 10
        self._WINDOW_WIDTH = 5
        self.urDb = urDb

    def show_basic_data(self, useEsc50=False):
        """
        Plot Linear-frequency power spectrogram for audio files

        Args:
            useEsc50: Flag if working with Esc50 dataset, will not be used! (once impelemented ESC-10 db class)
        """
        if useEsc50:
            print("Showing esc50 dataset")
            dat1, sampling_rate1 = librosa.load(
                self.urDb.BASE_PATH + "//audio//1-34497-A-14.wav")
            dat2, sampling_rate2 = librosa.load(
                self.urDb.BASE_PATH + "//audio//1-50661-A-44.wav")
        else:
            dat1, sampling_rate1 = librosa.load(
                self.urDb.BASE_PATH + "//audio//fold5//100032-3-0-0.wav")
            dat2, sampling_rate2 = librosa.load(
                self.urDb.BASE_PATH + "//audio//fold5//100263-2-0-117.wav")
        plt.figure(figsize=(self._WINDOW_HEIGHT, self._WINDOW_WIDTH))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(dat1)), ref=np.max)
        plt.subplot(4, 2, 1)
        librosa.display.specshow(D, y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Linear-frequency power spectrogram')

        D = librosa.amplitude_to_db(np.abs(librosa.stft(dat2)), ref=np.max)
        plt.subplot(4, 2, 2)
        librosa.display.specshow(D, y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Linear-frequency power spectrogram')
        plt.show()

    def show_diff_classes(self):
        """
        Using random samples to observe difference in waveforms from Lin.-fq power spectrograms
        """
        arr = np.array(self.urDb.df["slice_file_name"])
        fold = np.array(self.urDb.df["fold"])
        cla = np.array(self.urDb.df["class"])

        j = 1
        plt.figure(figsize=(self._WINDOW_HEIGHT, self._WINDOW_WIDTH))
        for i in range(175, 197, 3):
            path = self.urDb.BASE_PATH + "//audio//fold" + \
                str(fold[i]) + '//' + arr[i]
            data, sampling_rate = librosa.load(path)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
            plt.subplot(4, 2, j)
            j = j + 1
            librosa.display.specshow(D, y_axis='linear')
            plt.colorbar(format='%+2.0f dB')
            plt.title(cla[i])
        plt.show()

    def plot_mel_spectrograms(self):
        """
        Function plots mel-spectogram of audio file
        """
        fig, axs = plt.subplots(4, 2, figsize=(10, 10))
        index = 0
        for col in range(2):
            for row in range(4):
                file_name = self.urDb.BASE_PATH + "//audio//fold" + \
                    str(self.urDb.df["fold"][index]) + '//' + \
                    self.urDb.df["slice_file_name"][index]
                audio_file, sr = librosa.load(file_name)
                audio_file = librosa.util.utils.fix_length(audio_file, 4*sr)

                S_mels = librosa.feature.melspectrogram(
                    y=audio_file, sr=sr, n_mels=self.urDb.IMG_HEIGHT, fmax=8000)
                print(S_mels.shape)
                print(S_mels.dtype)
                print(S_mels)
                S_dB = librosa.power_to_db(S_mels, ref=np.max)
                librosa.display.specshow(
                    S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=axs[row][col])
                axs[row][col].tick_params(axis='x', labelsize=7, pad=0)
                axs[row][col].set_xlabel('time, s', fontsize=8)
                axs[row][col].set_title('Mel Spectrogram of: {}'.format(
                    self.urDb.df["class"][index]), fontsize=12, pad=0)
                index += 60  # get some differerent sounds
        plt.tight_layout()
        plt.show()
            
    def plot_mffc(self):
        fig, axs = plt.subplots(4, 2, figsize=(10, 10))
        index = 0
        for col in range(2):
            for row in range(4):
                file_name = self.urDb.BASE_PATH + "//audio//fold" + \
                    str(self.urDb.df["fold"][index]) + '//' + \
                    self.urDb.df["slice_file_name"][index]
                audio_signal, sr = librosa.load(file_name)
                audio_signal = librosa.util.utils.fix_length(audio_signal, 4*sr)
                mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=40, hop_length=1024)
                # hop_length width of resulting array
                
                librosa.display.specshow(mfccs, 
                         x_axis="time", y_axis='frames',
                         sr=sr, ax=axs[row][col])
                axs[row][col].set_xlabel('Laikas, s', fontsize=8)
                axs[row][col].set_ylabel('Koeficientai', fontsize=8)
                axs[row][col].set_title('MFCC of: {}'.format(
                    self.urDb.df["class"][index]), fontsize=12, pad=0)
                index += 60  # get some differerent sounds
        plt.tight_layout()
        plt.show()  

    def plot_wave_from_audio(self):
        """ 
        Function plots audio file in wave form
        """
        fig, axs = plt.subplots(4, 2, figsize=(10, 10))
        index = 0
        for col in range(2):
            for row in range(4):
                path = self.urDb.BASE_PATH + "//audio//fold" + \
                    str(self.urDb.df["fold"][index]) + '//' + \
                    self.urDb.df["slice_file_name"][index]
                data, sr = librosa.load(path)
                data = librosa.util.utils.fix_length(data, 4*sr)

                librosa.display.waveshow(
                    data, sr=sr, x_axis='time', ax=axs[row][col], offset=0.0, marker='', where='post')
                axs[row][col].set_ylim(-1, 1)
                axs[row][col].set_title(
                    "Audio signal in wave form of: " + str(self.urDb.df["class"][index]), fontsize=12, pad=0)
                index += 60  # get some differerent sounds
        plt.tight_layout()
        plt.show()

    def plot_basic_spectrograms(self):
        """ 
        Function calculates STFT for audio file and plots spectrogram
        """
        FRAME_SIZE = 2048
        HOP_SIZE = 512

        fig, axs = plt.subplots(4, 2, figsize=(10, 10))
        index = 0
        for col in range(2):
            for row in range(4):
                file_name = self.urDb.BASE_PATH + "//audio//fold" + \
                    str(self.urDb.df["fold"][index]) + '//' + \
                    self.urDb.df["slice_file_name"][index]

                audio_file, sample_rate = librosa.load(file_name)
                audio_file = librosa.util.utils.fix_length(
                    audio_file, 4*sample_rate)

                stft = librosa.stft(
                    audio_file, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)  # STFT of y
                S_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
                librosa.display.specshow(S_db,
                                         sr=sample_rate,
                                         hop_length=HOP_SIZE,
                                         x_axis="time",
                                         y_axis='log',
                                         ax=axs[row][col])
                axs[row][col].tick_params(axis='x', labelsize=7, pad=0)
                axs[row][col].set_xlabel('time, s', fontsize=8)
                axs[row][col].set_title('Spectrogram of: {}'.format(
                    self.urDb.df["class"][index]), fontsize=12, pad=0)
                index += 60  # get some differerent sounds
        plt.tight_layout()
        plt.show()

    def plot_history_results(self, history):
        plt.subplot(121)
        plt.plot(history['accuracy'], 'r')
        plt.plot(history['val_accuracy'], 'b')
        plt.ylabel('accuracy, r - train, b - val')
        plt.xlabel('epoch')
        plt.grid(b=True)

        plt.subplot(122)
        plt.plot(history['loss'], 'r')
        plt.plot(history['val_loss'], 'b')
        plt.ylabel('Loss, r - train, b - val')
        plt.xlabel('epoch')
        plt.grid(b=True)
        plt.show()

    def plot_to_compare_models(self, history_1, history_2):
        plt.subplot(121)
        plt.plot(history_1['accuracy'], 'r')
        plt.plot(history_2['accuracy'], 'r--')
        plt.plot(history_1['val_accuracy'], 'b')
        plt.plot(history_2['val_accuracy'], 'g')
        plt.ylabel('Tikslumas, raudona - treniravimo, žalia/mėlyna - testavimo')
        plt.xlabel('Epocha')
        plt.grid(b=True)

        plt.subplot(122)
        plt.plot(history_1['loss'], 'r')
        plt.plot(history_1['val_loss'], 'b')
        plt.plot(history_2['loss'], 'r--')
        plt.plot(history_2['val_loss'], 'g')
        plt.ylabel('Nuostolis (angl. loss), raudona - treniravimo, žalia/mėlyna - testavimo')
        plt.xlabel('Epocha')
        plt.grid(b=True)
        plt.show()    

def main():

    print("Hello from Visualise!")
    vis = Visualise()
    # vis.plot_basic_spectrograms()
    # vis.plot_wave_from_audio()
    # vis.plot_mel_spectrograms()
    # vis.plot_mffc()


    hist=np.load('overfit_model/my_history_overfit.npy',allow_pickle='TRUE').item()
    # vis.plot_history_results(hist)

    hist2=np.load('overfit_model/first_modGood-noSmoot/my_history.npy',allow_pickle='TRUE').item()
    # vis.plot_history_results(hist2)

    vis.plot_to_compare_models(hist, hist2)


    print("End from Visualise!")


if __name__ == "__main__":
    main()
