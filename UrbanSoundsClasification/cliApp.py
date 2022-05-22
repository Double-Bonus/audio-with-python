# https://thegeekpage.com/stereo-mix/

from sys import byteorder
from array import array
from struct import pack

import pyaudio, wave, librosa
import os, cv2
import numpy as np
from functionality import Functionality
from tensorflow import keras

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100


IMG_HEIGHT = 128
IMG_WIDTH = 173 # 88.200 / hopsize = 512


class_names = [
'Kondicionierius',
'Mašinos signalas',
'Vaikų žaidimai',
'Šuns lojimas',
'Gręžimas',
'Variklio darbas',
'Ginklo šūvis',
'Skaldymo kūjis',
'Sirena',
'Muzika',
'10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
'31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49'
] #todo add proper names!


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE,
        input_device_index = 0)
    
    r = array('h')
    import time
    t_end = time.time() + 5
    # while 1:
    while time.time() < t_end:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)


    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
########################################################### end of recording part


def wav_to_sprectrgoram():
        file_name = 'demo.wav'

        y, sr = librosa.load(file_name, res_type='kaiser_fast') 
        
        img_name = 'testImg' + '.png'
        hop_length = 512           # number of samples per time-step in spectrogram
        n_mels = IMG_HEIGHT        # number of bins in spectrogram. Height of image
        time_steps = IMG_WIDTH - 1 # number of time-steps. Width of image
        
        y = librosa.util.utils.fix_length(y, 88200)
        
        length_samples = time_steps * hop_length
        window = y[0:length_samples]

        dir_name = "testImg"
        spectrogram_image(y=window, sr=sr, out_dir=dir_name , out_name=img_name, hop_length=hop_length, n_mels=n_mels)


def spectrogram_image(y, sr, out_dir, out_name, hop_length, n_mels):
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
    spec = np.log(spec + 1e-9) # add small number to avoid log(0)
        
    # min-max scale to fit inside 8-bit range
    img = Functionality.scale_minmax(spec, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255 - img            # invert. make black==more energy

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    cv2.imwrite((out_dir + "\\" + out_name), img) # save as PNG


def load_spectogram():
    imagesArr = np.zeros((2, IMG_HEIGHT, IMG_WIDTH))

    image_path = "testImg//testImg.png"
    image= cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    if image is None:
        print("Error, image was not found from: " + image_path)
        quit()
    image = np.array(image)
    image = image.astype('float32')
    image /= 255
    imagesArr[0] = image 
    return imagesArr


def yes_or_no():
    answer = input("Klausytis vėl? (y/n):  ")
    if answer == "y": 
        return True 
    elif answer == "n": 
        return False 
    else: 
        print("Please enter yes or no.")
        return False 



if __name__ == '__main__':
    print("Audio clasifikatoriaus programa")
    
    print("Paleidziamas modelis")
    model = keras.models.load_model('models\\esc50.h5')
    print("Modelis paleistas")
    
    appState = True
    while appState:
        print("Klausomasi garso signalo")
        record_to_file('demo.wav')
        wav_to_sprectrgoram()
        
        x_test = load_spectogram()
        x_test = x_test.reshape(x_test.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
        
        
        pred = model.predict(x_test)

        y_pred = np.argmax(pred, axis=1) 


        print("Modelio sprendimas: " + class_names[y_pred[0]])
        print("Sprendimo tikimybe: " + str(pred[0][y_pred[0]]))
        
        appState = yes_or_no()
        print("\n")
    
    print("Baigta programa")