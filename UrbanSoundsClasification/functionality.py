import librosa
import numpy as np
import os
import cv2
import sklearn
from sklearn.metrics import accuracy_score, f1_score

from tensorflow import keras
from tensorflow.keras import callbacks



from visualise import *
from cnn_model import *


class Functionality:
    def calculate_accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def calculate_F1score(y_true, y_pred, verbose = False):
        av_none = f1_score(y_true, y_pred, average=None)
        av_micro = f1_score(y_true, y_pred, average='micro')
        av_macro = f1_score(y_true, y_pred, average='macro')
        av_weighted = f1_score(y_true, y_pred, average='weighted')

        if verbose:
            print("Using None average")
            print(av_none)
            print(np.mean(av_none))

            print("Using micro average")
            print(av_micro)

            print("Using macro average")
            print(av_macro)

            print("Using weighted average")
            print(av_weighted)
        return av_macro, av_weighted

    
    def scale_minmax(X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled


def save_wav_to_png(df, DATA_SAMPLES_CNT, BASE_PATH, IMG_HEIGHT, IMG_WIDTH, use_Kfold = False):
    """ 
    Saves spectograms data from sound files as png pictures
    """
    print("Saving pictures to drive")
    for i in range(DATA_SAMPLES_CNT):
        file_name = BASE_PATH  + "//audio//" + str(df["filename"][i])
        # Here kaiser_fast is a technique used for faster extraction
        y, sr = librosa.load(file_name, res_type='kaiser_fast') 
        
        img_name = 'out' + str(i+1) + "_" + str(df["target"][i]) + '.png'
        hop_length = 512           # number of samples per time-step in spectrogram
        n_mels = IMG_HEIGHT        # number of bins in spectrogram. Height of image
        time_steps = IMG_WIDTH - 1 # number of time-steps. Width of image (TODO FIX it add 1 px to width!!)
        
        
        # sr * 4 = size!!!
        # 22050 * 4 = 88200
        
        
        y = librosa.util.utils.fix_length(y, sr * 5)  ## suppose it works????? It just adds white space!!!!
        
        start_sample = 0 # starting at beginning
        length_samples = time_steps * hop_length
        window = y[start_sample:start_sample+length_samples]
        
        if use_Kfold:
            dir_name = "processed//fold" + str(df["fold"][i])
        else:
            dir_name = "img_esc"
        
        spectrogram_image(y=window, sr=sr, out_dir=dir_name , out_name=img_name, hop_length=hop_length, n_mels=n_mels)
    print("Done saving pictures!")
    
      
def spectrogram_image(y, sr, out_dir, out_name, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
    # mels = librosa.feature.melspectrogram(y=y, sr=sr)
    
    if 1:
        mels = np.log(mels + 1e-9) # add small number to avoid log(0)
    else:  #testing !
        mels = np.mean(mels, axis=0)

    # min-max scale to fit inside 8-bit range
    img = Functionality.scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255 - img            # invert. make black==more energy

    # save as PNG
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    cv2.imwrite((out_dir + "\\" + out_name), img)
    
def load_spectograms(df, DATA_SAMPLES_CNT, IMG_HEIGHT, IMG_WIDTH):
    """ 
    Loads images to RAM from folder.

    Returns:
        Images arrau and class_name for y values
    """
    print("Loading images from drive to RAM!")
    img_data_array = np.zeros((DATA_SAMPLES_CNT, IMG_HEIGHT, IMG_WIDTH))
    class_name = np.zeros((DATA_SAMPLES_CNT, 1))
    
    cla = np.array(df["target"]) # TODO FIX suppose its global

    for i in range(0, DATA_SAMPLES_CNT):
        image_path = "img_esc//" + "out" + str(i+1) + "_" + str(df["target"][i]) + ".png"
        image= cv2.imread(image_path, cv2.COLOR_BGR2RGB) # TODO FIX: check color map
        # image= cv2.imread(image_path)
        if image is None:
            print("Error, image was not found from: " + image_path)
            quit()
        # image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
        image = np.array(image)
        image = image.astype('float32')
        image /= 255
        img_data_array[i] = image 
        class_name[i] = cla[i]
    print("Finish loading images from drive to RAM!")
    
    return img_data_array, class_name



def train_CNN(X, Y, IMG_HEIGHT, IMG_WIDTH, CLASSES_CNT, test_portion = 0.25):
    """ 
    Trains CNN with givrn inputs and predifend image dimensions

    Args:
        X : data inputs
        Y : data outputs
        test_portion (float, optional): What portion of data is used to validate. Defaults to 0.25.
    """
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=test_portion, random_state=7)
    
    x_train = x_train.reshape(x_train.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
    x_test = x_test.reshape(x_test.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
    
    train_labels = keras.utils.to_categorical(y_train, num_classes=CLASSES_CNT)
    test_labels = keras.utils.to_categorical(y_test, num_classes=CLASSES_CNT)
    
    model = get_cnn_minKernel_smallerL2_2(IMG_HEIGHT, IMG_WIDTH, CLASSES_CNT)
    model.summary()

    epochCnt = 400
    earlystopper = callbacks.EarlyStopping(patience=epochCnt*0.2, verbose=1, monitor='val_accuracy')
    checkpointer = callbacks.ModelCheckpoint('models\\esc10.h5', verbose=1, monitor='val_accuracy', save_best_only=True)
        
    hist = model.fit(x_train, train_labels, batch_size=32, epochs=epochCnt, verbose=1, 
    validation_data=(x_test, test_labels), callbacks = [earlystopper, checkpointer])
    draw_model_results(hist)
    log_confusion_matrix(model, x_test, y_test) #TODO FIX: Note that here you use last model not the one saved!





if __name__ == "__main__":
    print("Runnnin functionaly")
