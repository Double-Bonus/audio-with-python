# https://docs.seldon.io/projects/alibi/en/latest/examples/anchor_image_imagenet.html

import tensorflow as tf
import matplotlib
from tensorflow import keras


import matplotlib.pyplot as plt
import numpy as np
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

def load_spectograms_1st_Fold():
    tempWidth = 173
    tempHeight = 128
    
    print("Loading images from drive to RAM!")
    img_data_array = np.zeros((873, tempHeight, tempWidth))
    class_name = np.zeros((873, 1))
    
    cla = np.array(df["classID"])
    idx = 0

    # for i in range(0, DATA_SAMPLES_CNT):
    for i in range(0, 2000):
        if df["fold"][i] == 1: # load only from first fold
            image_path = "processed//fold1//out" + str(i+1) + "_" + str(df["class"][i]) + ".png"
            image= cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            if image is None:
                print("Error, image was not found from: " + image_path)
                quit()
            # image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            
            img_data_array[idx] = image 
            class_name[idx] = cla[i]  
            idx = idx + 1
    return img_data_array, class_name

def explainAlibi():
    # https://docs.seldon.io/projects/alibi/en/stable/examples/anchor_image_imagenet.html
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
        
def explainShapEXAMPLE():
   # https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/Explain%20ResNet50%20using%20the%20Partition%20explainer.html
    import json
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    import shap
    # load pre-trained model and data
    model = ResNet50(weights='imagenet')
    X, y = shap.datasets.imagenet50()

    # getting ImageNet 1000 class names
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    with open(shap.datasets.cache(url)) as file:
        class_names = [v[1] for v in json.load(file).values()]
    #print("Number of ImageNet classes:", len(class_names))
    #print("Class names:", class_names)
    # python function to get model output; replace this function with your own model function.
    def f(x):
        tmp = x.copy()
        preprocess_input(tmp)
        return model(tmp)
    
    
    print(X[0].shape)
    print(X[0].shape)
    print(X[0].shape)

    # define a masker that is used to mask out partitions of the input image.
    masker_blur = shap.maskers.Image("blur(128,128)", X[0].shape)
    
    
    
    # create an explainer with model and image masker
    explainer_blur = shap.Explainer(f, masker_blur, output_names=class_names)

    # here we explain two images using 500 evaluations of the underlying model to estimate the SHAP values
    shap_values_fine = explainer_blur(X[1:3], max_evals=5000, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])
    
    

    # output with shap values
    shap.image_plot(shap_values_fine)

def explainShap():
    import shap
    
    # model = keras.models.load_model('models/urban_model.h5') # imput (None, 128, 128, 1)
    model = keras.models.load_model('models/k_urban_model.h5') # imput (None, 128, 173, 1)
    X_data, Y_data = load_spectograms_1st_Fold()
    
    X_data = X_data.reshape(X_data.shape[0], 128, 173, 1)
    
    def make_predictions(x):
        # X_batch = vectorizer.transform(X_batch_text).toarray()
        preds = model.predict(x)
        return preds

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
    'Muzika']

    # define a masker that is used to mask out partitions of the input image.
    masker_blur = shap.maskers.Image("blur(128,128)", X_data[0].shape)
    
    
    # create an explainer with model and image masker
    explainer_blur = shap.Explainer(make_predictions, masker_blur, output_names = class_names)

    for i in range(20, 21): # number of images to explain * 3 (Note that this api takes long time to calculate)
        # here we explain images using 500 evaluations of the underlying model to estimate the SHAP values
        shap_values_fine = explainer_blur(X_data[i*3:(i+1)*3], max_evals=6000, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])
        print('Using image nr: ', i*3, '-', (i+1)*3)
        shap.image_plot(shap_values_fine)
    
def explainLime():
    import lime
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    import matplotlib.pyplot as plt
    import random   
    
    model = keras.models.load_model('models/k_urban_model.h5') # imput (None, 128, 173, 1)
    X_data, Y_data = load_spectograms_1st_Fold()
    
    # X_data = X_data.reshape(X_data.shape[0], 128, 173, 1)
    
    def new_predict_fn(images):
        # images = convert_to_1channel(images)
        gray = images[:,:,:,0]

        return model.predict(gray)
                         
                         
    print(X_data.shape)                       
                         
    explainer = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
            X_data[10], 
            new_predict_fn
    )
    plt.imshow(X_data[10])
    print(X_data.shape)
    X_data_pred = X_data.reshape(X_data.shape[0], 128, 173, 1)
    test = X_data_pred[8]
    
    
    print("\n\n\n\n\n")
    print(test.shape)
    print("\n\n\n\n\n")
    
    image, mask = explanation.get_image_and_mask(
            model.predict(test).argmax(axis=1)[0],
            positive_only=True, 
            hide_rest=False)
    plt.imshow(mark_boundaries(image, mask))
        
if 0: # Alibi api
    explainAlibi()
elif 0: # Lime TODO: FIX this
    explainLime()
else: # Shap
    explainShap()
    
    