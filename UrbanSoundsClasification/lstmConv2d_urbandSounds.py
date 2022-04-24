from tensorflow import keras
from keras import metrics, callbacks
import numpy as np
import pandas as pd
import sklearn.model_selection
import cv2, librosa, os

# user created libs:
from lstm_models import * 
from functionality import scale_minmax, get_class_weights

# User defined classes
from datasetsBase import UrbandSound8k



BASE_PATH = "Urband_sounds//UrbanSound8K"
DATA_SAMPLES_CNT = 8732
CLASSES_CNT = 10
FRAME_CNT = 8 # number of frames for lstm
TEST_PORTION = 0.25
IMG_HEIGHT = 64
IMG_WIDTH = 128
df = pd.read_csv("Urband_sounds//UrbanSound8K//metadata//UrbanSound8K.csv")

if IMG_WIDTH % FRAME_CNT != 0:
    print("ERROR: IMG_WIDTH must be divisible by FRAME_CNT")
    exit()


def spectrogram_image(y, sr, out_dir, out_name, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
    # mels = librosa.feature.melspectrogram(y=y, sr=sr)
    
    if 1:
        mels = np.log(mels + 1e-9) # add small number to avoid log(0)
    else:  #testing !
        mels = np.mean(mels, axis=0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255 - img            # invert. make black==more energy

    # save as PNG
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    cv2.imwrite((out_dir + "\\" + out_name), img)
    
def save_wav_to_png(use_Kfold = False):
    """ 
    Saves spectograms data from sound files as png pictures
    """
    print("Saving pictures to drive")
    for i in range(DATA_SAMPLES_CNT):
        file_name = BASE_PATH  + "//audio//fold" + str(df["fold"][i]) + '//' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        y, sr = librosa.load(file_name, res_type='kaiser_fast') 
        
        hop_length = 512           # number of samples per time-step in spectrogram
        n_mels = IMG_HEIGHT        # number of bins in spectrogram. Height of image
        time_steps = IMG_WIDTH - 1 # number of time-steps. Width of image (TODO FIX it add 1 px to width!!)
        
        
        # sr * 4 = size!!!
        # 22050 * 4 = 88200
        # y = librosa.util.utils.fix_length(y, 75000)  ## suppose it works????? It just adds white space!!!!
        y = librosa.util.utils.fix_length(y, 88200)  ## suppose it works????? It just adds white space!!!!
        # y, sr = pad_trunc(y, sr, 4000)
        
        start_sample = 0 # starting at beginning
        length_samples = time_steps * hop_length
        window = y[start_sample:start_sample+length_samples]
        
        if use_Kfold:
            dir_name = "processed//fold" + str(df["fold"][i])
        else:
            dir_name = "img_save_lstmCnn_f" + str(FRAME_CNT)
            
        if FRAME_CNT == 2:    
            a_win = window[:len(window)//FRAME_CNT]
            b_win = window[len(window)//FRAME_CNT:]
            img_name_a = 'out' + str(i+1) + "_" + str(df["class"][i]) + 'a' + '.png'
            img_name_b = 'out' + str(i+1) + "_" + str(df["class"][i]) + 'b' + '.png'
            spectrogram_image(y=a_win, sr=sr, out_dir=dir_name , out_name=img_name_a, hop_length=hop_length, n_mels=n_mels)
            spectrogram_image(y=b_win, sr=sr, out_dir=dir_name , out_name=img_name_b, hop_length=hop_length, n_mels=n_mels)
        elif FRAME_CNT == 8:
            imgParts = np.array_split(window, FRAME_CNT)
            names = ["a", "b", "c", "d", "e", "f", "g", "h"]
            indx = 0
            for part in imgParts:
                part_name = "out" + str(i+1) + "_" + str(df["class"][i]) + names[indx] + ".png"
                spectrogram_image(y=part, sr=sr, out_dir=dir_name , out_name=part_name, hop_length=hop_length, n_mels=n_mels)
                indx = indx + 1
        else:
            print("ERROR: FRAME_CNT must be 2 or 8")
            exit()    
            
            
            
            
    print("Done saving pictures!")

def load_spectograms():
    """ 
    Loads images to RAM from folder.

    Returns:
        Images arrau and class_name for y values
    """
    print("Loading images from drive to RAM!")
    img_data_array = np.zeros((DATA_SAMPLES_CNT, FRAME_CNT, IMG_HEIGHT, (IMG_WIDTH // FRAME_CNT)))
    class_name = np.zeros((DATA_SAMPLES_CNT, 1))
    
    cla = np.array(df["classID"]) # TODO: FIX suppose its global
    names = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for i in range(0, DATA_SAMPLES_CNT):
        for j in range(0, FRAME_CNT): # number of images to lstm
            ext = names[j]
            image_path = "img_save_lstmCnn_f" + str(FRAME_CNT) + "//out" + str(i+1) + "_" + str(df["class"][i]) + ext + ".png"
            image= cv2.imread(image_path, cv2.COLOR_BGR2RGB) # TODO FIX: check color map
            # image= cv2.imread(image_path)
            if image is None:
                print("Error, image was not found from: " + image_path)
                quit()
            # image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array[i][j] = image 
            class_name[i] = cla[i]  
    return img_data_array, class_name

def train(x_train, y_train, x_test, y_test, num_classes, epochs, verbose = False):

    # Customize and print x & y shapes
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    print(' ')
    print('Shapes: x_train {}, x_test {}, y_train {}, y_test {}'.format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    cnn, lstm = get_lstm(x_train, num_classes, 13)

    if verbose:
        # summarize model
        try:        cnn.summary()
        except:     print(" ")
        finally:    lstm.summary()


    earlystopper = callbacks.EarlyStopping(patience=epochs*0.35, verbose=1, monitor='val_accuracy')
    checkpointer = callbacks.ModelCheckpoint('models\\lstmModel.h5', verbose=1, monitor='val_accuracy', save_best_only=True)

    # compile, fit, evaluate
    lstm.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics = ['accuracy'])

    if 1: # use class wieghts
        clsWeight = get_class_weights()
        lstm.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=0, validation_data=(x_test, y_test), class_weight = clsWeight,
            callbacks=[earlystopper, checkpointer])
    else:
        lstm.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=0, validation_data=(x_test, y_test), 
            callbacks=[earlystopper, checkpointer])
    score = lstm.evaluate(x_test, y_test, verbose=1) 
    return score[1]  # {'loss': 0.2, 'accuracy': 0.7}.
    
    
    
    
    
################################################################
if not os.path.exists("img_save_lstmCnn_f" + str(FRAME_CNT)):
    save_wav_to_png()
    
X_data, Y_data = load_spectograms()

print("The size of a numpy array is: ", X_data.shape)
print("The size of a numpy array is: ", X_data[0].shape)

if 1: #kfold
    urbandDb = UrbandSound8k(IMG_HEIGHT, IMG_WIDTH) # create class instance for dataset
    urbandDb.set_frame_cnt(FRAME_CNT)
    
    accuracies = np.zeros((urbandDb.CLASSES_CNT, 1))
    
    folds_cnt = np.zeros(urbandDb.CLASSES_CNT, dtype=int)
    for i in range(0, urbandDb.DATA_SAMPLES_CNT):
         folds_cnt[df["fold"][i] -1 ]  =  folds_cnt[df["fold"][i] -1] + 1

    # 10-fold cross validation
    kfoldsCnt = 10
    for test_index in range(0, kfoldsCnt):
        x_train, x_test, y_train, y_test = urbandDb.prepare_data_kFold_LSTM(test_index, kfoldsCnt, folds_cnt, X_data, Y_data)
        accuracies[test_index] = train(x_train, y_train, x_test, y_test, CLASSES_CNT, epochs=150, verbose=(0 == test_index))
        print("Temp k-Folds Accuracy: {0}".format(np.mean(accuracies)))
        
    print("Average 10 Folds Accuracy: {0}".format(np.mean(accuracies)))
    print("Standart deviation of accuracy: {0}".format(np.std(accuracies)))

else:
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X_data, Y_data, test_size=TEST_PORTION, random_state=7)
    
    train(x_train, y_train, x_test, y_test, CLASSES_CNT, epochs=150)
