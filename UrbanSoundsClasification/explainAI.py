# https://docs.seldon.io/projects/alibi/en/latest/examples/anchor_image_imagenet.html

import tensorflow as tf
import matplotlib
from tensorflow import keras


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from alibi.datasets import load_cats
from alibi.explainers import AnchorImage
import cv2
import pandas as pd


IMG_HEIGHT = 64
IMG_WIDTH = 128
DATA_SAMPLES_CNT = 8732
Start_Sample = 0 # ofset if I'm using all images
df = pd.read_csv("Urband_sounds//UrbanSound8K//metadata//UrbanSound8K.csv")


def load_spectograms():
    """ 
    Loads images to RAM from folder.

    Returns:
        Images arrau and class_name for y values
    """
    print("Loading images from drive to RAM!")
    img_data_array = np.zeros((DATA_SAMPLES_CNT, IMG_HEIGHT, IMG_WIDTH)) # some how it adds one pixcel
    class_name = np.zeros((DATA_SAMPLES_CNT, 1))
    
    cla = np.array(df["classID"]) # TODO FIX suppose its global

    for i in range(0, DATA_SAMPLES_CNT):
        i = i + Start_Sample
        image_path = "img_save//" + "out" + str(i+1) + "_" + str(df["class"][i]) + ".png"
        image= cv2.imread(image_path, cv2.COLOR_BGR2RGB) # TODO FIX: check color map
        # image= cv2.imread(image_path)
        if image is None:
            print("Error, image was not found from: " + image_path)
            quit()
        # image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
        image = np.array(image)
        image = image.astype('float32')
        image /= 255
        
        i = i - Start_Sample
        img_data_array[i] = image 
        class_name[i] = cla[i]  
    return img_data_array, class_name

def testAi():
    model = InceptionV3(weights='imagenet')

    model = keras.models.load_model('models/urban_model.h5')
    
    image_shape = (IMG_HEIGHT, IMG_WIDTH, 1)
    X_data, Y_data = load_spectograms()
    X_data = X_data.reshape(X_data.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)
    
    print(f'Images shape: {X_data.shape}')

    preds = model.predict(X_data)
    # label = decode_predictions(preds, top=3)
    # print(label[0])

    predict_fn = lambda x: model.predict(x)

    segmentation_fn = 'slic'
    kwargs = {'n_segments': 15, 'compactness': 20, 'sigma': .5}
    explainer = AnchorImage(predict_fn, image_shape, segmentation_fn=segmentation_fn,
                            segmentation_kwargs=kwargs, images_background=None)

    for i in range(0, DATA_SAMPLES_CNT):
        plt.imshow(X_data[i]);

        image = X_data[i]
        np.random.seed(0)
        print(image.shape)
        
        explanation = explainer.explain(image, threshold=.95, p_sample=.5, tau=0.25)
        
        # plt.imshow(explanation.anchor);
        # plt.show()
        # plt.imshow(explanation.segments);
        # plt.show()
        plt.title("Model explanation using 'ALIBI EXPLAIN' API of audio nr: " + str(i + 1))
        plt.imshow(explanation.anchor);
        savePath_a = 'XaiRes/explainAudio_' + str(i + 1) + '_a' + '.png'
        plt.savefig(savePath_a)
        
        plt.imshow(explanation.segments);
        savePath_b = 'XaiRes/explainAudio_' + str(i + 1) + '_b' + '.png'
        plt.savefig(savePath_b)
        
        
        

testAi()
