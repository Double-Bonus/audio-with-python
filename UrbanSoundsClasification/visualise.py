# File for handling plots and visualising data

import sklearn
import matplotlib.pyplot as plt
import numpy as np
import itertools
import librosa

# ----------------------------- Private Defines ---------------------------------
_WINDOW_HEIGHT = 10
_WINDOW_WIDTH = 5


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
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):  # TODO why need itertools???
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
  plt.ylabel('accuracy, r - train, b - val')
  plt.xlabel('epoch')
  plt.grid(b=True)
  
  plt.subplot(122)
  plt.plot(model_history.history['loss'], 'r')
  plt.plot(model_history.history['val_loss'], 'b')
  plt.ylabel('Loss, r - train, b - val')
  plt.xlabel('epoch')
  plt.grid(b=True)
  plt.show()
    
def show_basic_data(base_path, useEsc50 = False):
  """
  Plot Linear-frequency power spectrogram for audio files

  Args:
      base_path : path to UrbandSounds8K db
      useEsc50: Flag if working with Esc50 dataset
  """
  if useEsc50:
    print("Showing esc50 dataset")
    dat1, sampling_rate1 = librosa.load(base_path + "//audio//1-34497-A-14.wav")
    dat2, sampling_rate2 = librosa.load(base_path + "//audio//1-50661-A-44.wav")
  else:
    dat1, sampling_rate1 = librosa.load(base_path + "//audio//fold5//100032-3-0-0.wav")
    dat2, sampling_rate2 = librosa.load(base_path + "//audio//fold5//100263-2-0-117.wav")
  plt.figure(figsize=(_WINDOW_HEIGHT, _WINDOW_WIDTH))
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

def show_diff_classes(df, base_path):
  """
  Using random samples to observe difference in waveforms from Lin.-fq power spectrograms

  Args:
      df : metadata frame for UrbandSounds8K
      base_path : path to UrbandSounds8K db
  """
  arr = np.array(df["slice_file_name"])
  fold = np.array(df["fold"])
  cla = np.array(df["class"])

  j = 1
  plt.figure(figsize=(_WINDOW_HEIGHT, _WINDOW_WIDTH))
  for i in range(175, 197, 3):
      path = base_path  + "//audio//fold" + str(fold[i]) + '//' + arr[i]
      data, sampling_rate = librosa.load(path)
      D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
      plt.subplot(4, 2, j)
      j = j + 1
      librosa.display.specshow(D, y_axis='linear')
      plt.colorbar(format='%+2.0f dB')
      plt.title(cla[i])
  plt.show()

def show_mel_img(base_path, img_h):  
  """
  Function shows mel-spectogram of audio file

  Args:
      base_path: path to UrbandSounds8K db
  """
  y, sr = librosa.load(base_path + "//audio//fold2//100652-3-0-0.wav")
  S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=img_h, fmax=8000)
  S_dB = librosa.power_to_db(S, ref=np.max)
  img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr, fmax=8000)
  plt.colorbar(format='%+2.0f dB')
  plt.title('Mel spectrogram')
  plt.show()

def plot_wave_from_audio(df, base_path):
  """ 
  Function plots audio file in wave form

  Args:
      df : metadata frame for UrbandSounds8K
      base_path : path to UrbandSounds8K db
  """
  for j in range(1, 3):
    i = np.random.randint(0, 8732)
    path = base_path  + "//audio//fold" + str(df["fold"][i]) + '//' + df["slice_file_name"][i]
    data, sr = librosa.load(path) 
    plt.subplot(2, 1, j)   
    librosa.display.waveshow(data, sr=sr, x_axis='time', offset=0.0, marker='', where='post')
    plt.title("Audio signal in wave form of: " + str(df["class"][i]))
  plt.show()