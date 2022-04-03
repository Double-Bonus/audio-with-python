from tensorflow import keras
from keras import metrics, callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import TimeDistributed, LSTM, ConvLSTM2D
import numpy as np
import pandas as pd
import sklearn.model_selection
import cv2, librosa, os



BASE_PATH = "Urband_sounds//UrbanSound8K"
DATA_SAMPLES_CNT = 8732
CLASSES_CNT = 10
TEST_PORTION = 0.25
IMG_HEIGHT = 64
IMG_WIDTH = 128
df = pd.read_csv("Urband_sounds//UrbanSound8K//metadata//UrbanSound8K.csv")



def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


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
            dir_name = "img_save_lstmCnn"
            
        a_win = window[:len(window)//2]
        b_win = window[len(window)//2:]
        img_name_a = 'out' + str(i+1) + "_" + str(df["class"][i]) + 'a' + '.png'
        img_name_b = 'out' + str(i+1) + "_" + str(df["class"][i]) + 'b' + '.png'
        
        
        spectrogram_image(y=a_win, sr=sr, out_dir=dir_name , out_name=img_name_a, hop_length=hop_length, n_mels=n_mels)
        spectrogram_image(y=b_win, sr=sr, out_dir=dir_name , out_name=img_name_b, hop_length=hop_length, n_mels=n_mels)
    print("Done saving pictures!")















def load_spectograms():
    """ 
    Loads images to RAM from folder.

    Returns:
        Images arrau and class_name for y values
    """
    print("Loading images from drive to RAM!")
    img_data_array = np.zeros((DATA_SAMPLES_CNT, 2, IMG_HEIGHT, (IMG_WIDTH // 2)))
    # img_data_array = np.zeros((DATA_SAMPLES_CNT, 2, IMG_HEIGHT, (IMG_WIDTH / 2))) # TODO: AR TAIP???
    class_name = np.zeros((DATA_SAMPLES_CNT, 1))
    
    cla = np.array(df["classID"]) # TODO: FIX suppose its global

    for i in range(0, DATA_SAMPLES_CNT):
        for j in range(0, 2): # number of images to lstm
            if j == 0: # TODO: this is retarded :)
                ext = "a"
            else:
                ext = "b"
            image_path = "img_save_lstmCnn//" + "out" + str(i+1) + "_" + str(df["class"][i]) + ext + ".png"
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

def train(x_train, y_train, x_test, y_test, num_classes, epochs):

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

    stage = 1
    if stage == 0:
        lstm = Sequential()
        lstm.add(ConvLSTM2D(8, kernel_size=(3, 3), strides=(1, 1), activation='elu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(MaxPooling2D(pool_size=(2, 3)))
        lstm.add(Dropout(0.25))
        lstm.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        lstm.add(MaxPooling2D(pool_size=(2, 2)))
        lstm.add(Dropout(0.25))
        lstm.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        lstm.add(MaxPooling2D(pool_size=(2, 2)))
        lstm.add(Dropout(0.25))
        lstm.add(Flatten())
        lstm.add(Dense(32, activation='elu'))
        lstm.add(Dense(num_classes, activation='softmax'))
    elif stage == 1:
        cnn = Sequential()
        cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(MaxPooling2D(pool_size=(1, 1)))
        cnn.add(Dropout(0.25))
        lstm = Sequential()
        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(ConvLSTM2D(16, kernel_size=(3, 3), strides=(1, 1), dropout=0.25, recurrent_dropout=0.25))
        lstm.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        lstm.add(MaxPooling2D(pool_size=(1, 2)))
        lstm.add(Dropout(0.25))
        lstm.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        lstm.add(MaxPooling2D(pool_size=(2, 2)))
        lstm.add(Dropout(0.25))
        lstm.add(Flatten())
        lstm.add(Dense(num_classes, activation='softmax'))
    else:
        cnn = Sequential()
        cnn.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='elu', input_shape=(x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=(2, 3)))
        cnn.add(Dropout(0.25))
        cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Dropout(0.25))
        cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Dropout(0.25))
        cnn.add(Flatten())

        lstm = Sequential()
        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, dropout=0.0, recurrent_dropout=0.0))
        lstm.add(Dense(num_classes, activation='softmax'))

    # summarize model
    try:        cnn.summary()
    except:     print(" ")
    finally:    lstm.summary()

    # compile, fit, evaluate
    lstm.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics = ['accuracy'])
    lstm.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[callbacks.EarlyStopping(monitor='val_accuracy', patience=epochs*0.2, restore_best_weights=True)])
    score = lstm.evaluate(x_test, y_test, verbose=1)

    # save model
    '''model_json = lstm.to_json()
    with open("Models\\model_cnn2dlstm.json", "w") as json_file:
        #
        json_file.write(model_json)
    lstm.save_weights("Models\\model_cnn2dlstm.h5")'''

    print(' ')

    # return score[1]
    
    
    
    
    
################################################################
if not os.path.exists("img_save_lstmCnn"):
    save_wav_to_png()
    
X_data, Y_data = load_spectograms()
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X_data, Y_data, test_size=0.25, random_state=7)
    
train(x_train, y_train, x_test, y_test, CLASSES_CNT, epochs=100)
