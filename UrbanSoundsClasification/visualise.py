# File for handling plots and visualising data



import sklearn
import matplotlib.pyplot as plt
import numpy as np
import itertools
import librosa



# ----------------------------- Private functions -------------------------------
# TODO FIX comment style you broken C style boi 
""" Returns a matplotlib figure containing the plotted confusion matrix.

Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
"""
def plot_confusion_matrix(cm, class_names):
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
  plot_confusion_matrix(cm, class_names=class_names)
  
  
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
    
    
def show_basic_data(BASE_PATH):
    dat1, sampling_rate1 = librosa.load(BASE_PATH + "//audio//fold5//100032-3-0-0.wav")
    dat2, sampling_rate2 = librosa.load(BASE_PATH + "//audio//fold5//100263-2-0-117.wav")
    plt.figure(figsize=(20, 10))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(dat1)), ref=np.max)
    plt.subplot(4, 2, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    # plt.show()

    # plt.figure(figsize=(20, 10))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(dat2)), ref=np.max)
    plt.subplot(4, 2, 2)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.show()


def show_diff_classes(df, BASE_PATH):
    '''Using random samples to observe difference in waveforms.'''
    arr = np.array(df["slice_file_name"])
    fold = np.array(df["fold"])
    cla = np.array(df["class"])

    j = 1
    plt.figure(figsize=(10, 5))
    for i in range(175, 197, 3):
        path = BASE_PATH  + "//audio//fold" + str(fold[i]) + '//' + arr[i]
        data, sampling_rate = librosa.load(path)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
        plt.subplot(4, 2, j)
        j = j + 1
        librosa.display.specshow(D, y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title(cla[i])
    plt.show()