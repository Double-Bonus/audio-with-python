from statistics import mode
from cv2 import threshold
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import LeakyReLU, ELU
from tensorflow.keras import regularizers
import tensorflow_addons as tfa

# Acc precentagies here show acc of testing with 1-st fold and trainging with other 9 (2-10)
#####################################################


################### testing
from keras import backend as K
from keras.losses import binary_crossentropy
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true, y_pred))

def get_cnn_batchNorm(img_h, img_w, class_cnt):
    model = Sequential()
    
    # Layer 1
    model.add(Conv2D(filters=48, kernel_size=5, activation='relu', kernel_regularizer=regularizers.l2(1e-3), input_shape = (img_h, img_w, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((3, 3), strides=3))
    model.add(Dropout(0.1))


    # Layer 2
    model.add(Conv2D(filters=80, kernel_size=4, activation='relu', kernel_regularizer=regularizers.l2(1e-3), 
        padding='valid' ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.25))

    # Layer 3
    model.add(Conv2D(filters=96, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(1e-3),
        padding='valid' ))    
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.5))

    # Flatter - same, right?
    model.add(GlobalAveragePooling2D())
    
    # Layer 4
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5))

    model.add(Dense(class_cnt, activation = "softmax"))
    
    # model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['categorical_accuracy'])
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=bce_dice_loss, metrics=['accuracy'])
    return model

def get_cnn_overfitting(img_h, img_w, class_cnt):
    model = Sequential()

    # Layer 1
    model.add(Conv2D(filters=64, kernel_size=5, activation='relu', input_shape = (img_h, img_w, 1)))
    model.add(MaxPooling2D((3, 3),strides=3 ))

    # Layer 2
    model.add(Conv2D(filters=128, kernel_size=4, activation='relu', padding='valid' ))
    model.add(MaxPooling2D((2, 2), strides=2 ))

    # Layer 3
    model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding='valid' ))    
    model.add(MaxPooling2D((2, 2), strides=2 ))

    # Layer 4
    model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='valid' ))    
    model.add(MaxPooling2D((2, 2), strides=2 ))

    # connection
    model.add(GlobalAveragePooling2D())
    
    # Layer 4
    model.add(Dense(128, activation = "relu"))
    model.add(Dense(class_cnt, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


# acc ~78-80 1 fold
def get_cnn_minKernelReg778(img_h, img_w, class_cnt):
    model = Sequential()
    
    alpha = 0.01

    # Layer 1
    model.add(Conv2D(filters=64, kernel_size=5, kernel_regularizer=regularizers.l2(1e-3), input_shape = (img_h, img_w, 1)))
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((3, 3), strides=3))
    model.add(Dropout(0.1))

    # Layer 2
    model.add(Conv2D(filters=128, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), 
        padding='valid' ))
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.25))


    # Layer 3
    model.add(Conv2D(filters=256, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3),
        padding='valid' ))    
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.5))

    # Flatter - same, right?
    model.add(GlobalAveragePooling2D())
    
    # Layer 4
    model.add(Dense(128))
    model.add(LeakyReLU(alpha))

    model.add(Dropout(0.5))

    model.add(Dense(class_cnt, activation = "softmax"))
    
    # model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['categorical_accuracy'])
    # model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=bce_dice_loss, metrics=['accuracy'])
    return model

###########################################################################


def get_cnn_moreConv(img_h, img_w, class_cnt):
    model = Sequential()
    
    alpha = 0.01

    # Layer 1
    model.add(Conv2D(filters=64, kernel_size=5, kernel_regularizer=regularizers.l2(1e-3), input_shape = (img_h, img_w, 1)))
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((3, 3), strides=3))
    model.add(Dropout(0.1))

    # Layer 2
    model.add(Conv2D(filters=128, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), 
        padding='valid' ))
    model.add(LeakyReLU(alpha))
    model.add(Conv2D(filters=128, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), 
        padding='valid' ))
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.25))


    # Layer 3
    model.add(Conv2D(filters=256, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3),
        padding='valid' ))    
    model.add(LeakyReLU(alpha))
    model.add(Conv2D(filters=256, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3),
        padding='valid' ))    
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.5))

    # Flatter - same, right?
    model.add(GlobalAveragePooling2D())
    
    # Layer 4
    model.add(Dense(128))
    model.add(LeakyReLU(alpha))

    model.add(Dropout(0.5))

    model.add(Dense(class_cnt, activation = "softmax"))
    
    # model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['categorical_accuracy'])
    # model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=bce_dice_loss, metrics=['accuracy'])
    return model



















def get_cnn_minKernelReg_flaterr(img_h, img_w, class_cnt):
    model = Sequential()
    
    alpha = 0.01

    # Layer 1
    model.add(Conv2D(filters=32, kernel_size=5, kernel_regularizer=regularizers.l2(1e-3), input_shape = (img_h, img_w, 1)))
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((3, 3), strides=3))
    model.add(Dropout(0.1))

    # Layer 2
    model.add(Conv2D(filters=48, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), 
        padding='valid' ))
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.25))


    # Layer 3
    model.add(Conv2D(filters=64, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3),
        padding='valid' ))    
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.25))

    # Layer 3
    model.add(Conv2D(filters=78, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3),
        padding='same' ))    
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.25))


    # Layer 3
    model.add(Conv2D(filters=96, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3),
        padding='same' ))    
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.25))

    # Flatter - same, right?
    model.add(Flatten())
    
    # Layer 4
    model.add(Dense(128))
    model.add(LeakyReLU(alpha))
    model.add(Dropout(0.5))

    # Layer 4
    model.add(Dense(128))
    model.add(LeakyReLU(alpha))
    model.add(Dropout(0.5))

    model.add(Dense(class_cnt, activation = "softmax"))
    
    # model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['categorical_accuracy'])
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=bce_dice_loss, metrics=['accuracy'])
    return model


# acc 
def get_cnn_minKernelReg2633(img_h, img_w, class_cnt):
    model = Sequential()
    
    alpha = 0.01

    # Layer 1
    model.add(Conv2D(filters=64, kernel_size=5, kernel_regularizer=regularizers.l2(1e-3), input_shape = (img_h, img_w, 1)))
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((3, 3), strides=3))

    # Layer 2
    model.add(Conv2D(filters=128, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), 
        padding='valid' ))
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((2, 2), strides=2 ))


    # Layer 3
    model.add(Conv2D(filters=256, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3),
        padding='valid' ))    
    model.add(LeakyReLU(alpha))
    model.add(MaxPooling2D((2, 2), strides=2 ))



    # Flatter - same, right?
    model.add(GlobalAveragePooling2D())
    
    # Layer 4
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))

    # Layer 4
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))


    model.add(Dense(class_cnt, activation = "softmax"))
    
    # model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['categorical_accuracy'])
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    # model.compile(optimizer='adam', loss=bce_dice_loss, metrics=['accuracy'])
    return model




# 75-77
def get_cnn_minKernelReg1235(img_h, img_w, class_cnt):
    model = Sequential()

    # Layer 1
    model.add(Conv2D(filters=32, kernel_size=5, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (img_h, img_w, 1)))
    model.add(MaxPooling2D((3, 3),strides=3 ))

    # Layer 2
    model.add(Conv2D(filters=64, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.25))

    # Layer 3
    model.add(Conv2D(filters=96, kernel_size=3, activation='relu', padding='valid' ))    
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.25))

    # Flatter - same?
    model.add(GlobalAveragePooling2D())
    
    # Layer 4
    model.add(Dense(72, activation = "relu"))
    model.add(Dropout(0.5))

    model.add(Dense(class_cnt, activation = "softmax"))
    
    # model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['categorical_accuracy'])
    # model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


#acc 77-78
def get_cnn_minKernelReg1112(img_h, img_w, class_cnt):
    model = Sequential()
    #1 2 kernel_regularizer=regularizers.l2(1e-3

    # Layer 1
    model.add(Conv2D(filters=36, kernel_size=5, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (img_h, img_w, 1)))
    model.add(MaxPooling2D((3, 3),strides=3 ))

    # Layer 2
    model.add(Conv2D(filters=48, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
    model.add(MaxPooling2D((2, 2), strides=2 ))

    # Layer 3
    model.add(Conv2D(filters=60, kernel_size=3, activation='relu', padding='valid' ))    


    model.add(GlobalAveragePooling2D())
    
    # Layer 4
    model.add(Dense(72, activation = "relu"))
    # model.add(Dropout(0.5))


    model.add(Dense(class_cnt, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


######################################################
######################################################

#acc 74-77, try something new
def get_cnn_minKernelReg66(img_h, img_w, class_cnt):
    model = Sequential()

    # Layer 1
    model.add(Conv2D(filters=24, kernel_size=5, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (img_h, img_w, 1)))
    model.add(MaxPooling2D((3, 3),strides=3 ))

    # Layer 2
    model.add(Conv2D(filters=36, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
    model.add(MaxPooling2D((2, 2), strides=2 ))

    # Layer 3
    model.add(Conv2D(filters=48, kernel_size=3, activation='relu', padding='valid' ))    
    model.add(GlobalAveragePooling2D())
    
    # Layer 4
    model.add(Dense(60, activation = "relu"))
    model.add(Dropout(0.5))


    model.add(Dense(class_cnt, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model

#acc 77-78, try bigger, trains relly slow
def get_cnn_minKernelReg6556(img_h, img_w, class_cnt):
    model = Sequential()
    #1 2 kernel_regularizer=regularizers.l2(1e-3

    # Layer 1
    model.add(Conv2D(filters=36, kernel_size=5, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (img_h, img_w, 1)))
    model.add(MaxPooling2D((3, 3),strides=3 ))

    # Layer 2
    model.add(Conv2D(filters=48, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
    model.add(MaxPooling2D((2, 2), strides=2 ))

    # Layer 3
    model.add(Conv2D(filters=60, kernel_size=3, activation='relu', padding='valid' ))    
    model.add(GlobalAveragePooling2D())
    
    # Layer 4
    model.add(Dense(72, activation = "relu"))
    model.add(Dropout(0.5))


    model.add(Dense(class_cnt, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model

# acc 77, more filters, add dropout
def get_cnn_minKernelReg1244(img_h, img_w, class_cnt):
    model = Sequential()
    #1 2 kernel_regularizer=regularizers.l2(1e-3

    # Layer 1
    model.add(Conv2D(filters=48, kernel_size=7, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (img_h, img_w, 1)))
    model.add(MaxPooling2D((3, 3),strides=3 ))
    model.add(Dropout(0.25))


    # Layer 2
    model.add(Conv2D(filters=96, kernel_size=5, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.25))

    # Layer 3
    model.add(Conv2D(filters=96, kernel_size=4, activation='relu', padding='valid' ))    
    model.add(GlobalAveragePooling2D())
    
    # Layer 4
    model.add(Dense(96, activation = "relu"))
    model.add(Dropout(0.5))

    model.add(Dense(class_cnt, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model

# acc ~73-75     Change kernel_regularizer=regularizers.l2(1e-3) to kernel_regularizer=regularizers.l2(1e-4)
def get_cnn_minKernelReg_12(img_h, img_w, class_cnt):
    model = Sequential()

    # Layer 1
    model.add(Conv2D(filters=64, kernel_size=5, kernel_regularizer=regularizers.l2(1e-4), activation='relu', input_shape = (img_h, img_w, 1)))
    model.add(MaxPooling2D((3, 3),strides=3 ))

    # Layer 2
    model.add(Conv2D(filters=96, kernel_size=4, kernel_regularizer=regularizers.l2(1e-4), activation='relu', padding='valid' ))
    model.add(MaxPooling2D((2, 2), strides=2 ))

    # Layer 3
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='valid' ))    
    # model.add(MaxPooling2D((2, 2), strides=2 ))


    # Same as flatern?
    model.add(GlobalAveragePooling2D())
    
    # Layer 4
    model.add(Dense(100, activation = "relu"))
    model.add(Dropout(0.5))


    model.add(Dense(class_cnt, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model

#evenbigger but with dropout, acc 75
def get_cnn_minKernelReg466(img_h, img_w, class_cnt):
    model = Sequential()


    #1 2 kernel_regularizer=regularizers.l2(1e-3

    # Layer 1
    model.add(Conv2D(filters=36, kernel_size=5, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (img_h, img_w, 1)))
    model.add(MaxPooling2D((3, 3),strides=3 ))
    model.add(Dropout(0.25))


    # Layer 2
    model.add(Conv2D(filters=72, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.25))

    # Layer 3
    model.add(Conv2D(filters=96, kernel_size=3, activation='relu', padding='valid' ))    
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.25))

    
    # Layer 4
    model.add(Dense(72, activation = "relu"))
    model.add(Dropout(0.5))


    model.add(Dense(class_cnt, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


#try second dence with dropout, acc 74-76
def get_cnn_minKernelReg5445(img_h, img_w, class_cnt):
    model = Sequential()


    #1 2 kernel_regularizer=regularizers.l2(1e-3

    # Layer 1
    model.add(Conv2D(filters=36, kernel_size=5, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (img_h, img_w, 1)))
    model.add(MaxPooling2D((3, 3),strides=3 ))

    # Layer 2
    model.add(Conv2D(filters=48, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
    model.add(MaxPooling2D((2, 2), strides=2 ))

    # Layer 3
    model.add(Conv2D(filters=60, kernel_size=3, activation='relu', padding='valid' ))    
    model.add(GlobalAveragePooling2D())
    
    # Layer 4
    model.add(Dense(72, activation = "relu"))
    model.add(Dropout(0.5))

    # Layer 5
    model.add(Dense(72, activation = "relu"))
    model.add(Dropout(0.5))

    model.add(Dense(class_cnt, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


#try leaky, was 78 prety much same
def get_cnn_minKernelReg65(img_h, img_w, class_cnt):
    model = Sequential()
    # alpha = 0.01 #atleast 78
    # alpha = 0.001 #atleast 76
    alpha = 0.015 #atleast 76
    ac = LeakyReLU(alpha)


    #1 2 kernel_regularizer=regularizers.l2(1e-3

    # Layer 1
    model.add(Conv2D(filters=36, kernel_size=5, kernel_regularizer=regularizers.l2(1e-3), activation=ac, input_shape = (img_h, img_w, 1)))
    model.add(MaxPooling2D((3, 3),strides=3 ))

    # Layer 2
    model.add(Conv2D(filters=48, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation=ac, padding='valid' ))
    model.add(MaxPooling2D((2, 2), strides=2 ))

    # Layer 3
    model.add(Conv2D(filters=60, kernel_size=3, activation=ac, padding='valid' ))    
    model.add(GlobalAveragePooling2D())
    
    # Layer 4
    model.add(Dense(72, activation = ac))
    model.add(Dropout(0.5))


    model.add(Dense(class_cnt, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


# 76-78
def get_cnn_minKernelReg(img_h, img_w, class_cnt):
    model = Sequential()
    # alpha = 0.01 #atleast 78
    # alpha = 0.001 #atleast 76
    alpha = 0.01 
    ac = LeakyReLU(alpha)

    # Layer 1
    model.add(Conv2D(filters=64, kernel_size=7, kernel_regularizer=regularizers.l2(1e-3), activation=ac, input_shape = (img_h, img_w, 1)))
    model.add(MaxPooling2D((3, 3),strides=3 ))

    # Layer 2
    model.add(Conv2D(filters=84, kernel_size=5, kernel_regularizer=regularizers.l2(1e-3), activation=ac, padding='valid' ))
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.25))


    # Layer 3
    model.add(Conv2D(filters=96, kernel_size=4,  kernel_regularizer=regularizers.l2(1e-3), activation=ac, padding='valid' ))    
    model.add(MaxPooling2D((2, 2), strides=2 ))
    model.add(Dropout(0.25))


    # Layer 3
    model.add(Conv2D(filters=96, kernel_size=3, activation=ac, padding='valid' ))    
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.25))

    
    # Layer 4
    model.add(Dense(100, activation = ac))
    model.add(Dropout(0.5))


    model.add(Dense(class_cnt, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


# Acc 63 ???
def get_simple_cnn(img_h, img_w, class_cnt):
    # Initialize model
    model = Sequential()
    
    # Layer 1
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape = (img_h, img_w, 1), padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Layer 2
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same' ))
    model.add(MaxPooling2D((2, 2)))

        
    # Layer 3
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.2))
    
    # Layer 4

    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5))
    
    # Layer 5
    model.add(Flatten())
    model.add(Dense(class_cnt, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model


# Acc 64 -
def get_simple_cnn_2(img_h, img_w, class_cnt):
    # Initialize model
    model = Sequential()
    
    # Layer 1
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape = (img_h, img_w, 1), padding='same'))
    model.add(MaxPooling2D((2, 4)))

    # Layer 2
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same' ))
    model.add(MaxPooling2D((2, 4)))
    
    # Layer 3
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.2))
    
    # Layer 4
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))
    
    # Layer 5
    model.add(Flatten())
    model.add(Dense(class_cnt, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model

# acc - 66?
#try bigger kernel with zero padding
def get_cnn_minKernelReg2(img_h, img_w, class_cnt):
    kernelSz = (3, 3)
    model = Sequential()
    
    # Layer 1
    model.add(Conv2D(filters=64, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3), kernel_size=(5,5),
     activation='relu', input_shape = (img_h, img_w, 1), padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Layer 2
    model.add(Conv2D(filters=128, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3), kernel_size=kernelSz,
     activation='relu', padding='same' ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    
    # Layer 3
    model.add(Conv2D(filters=256, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3), kernel_size=kernelSz,
     activation='relu', padding='same' ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Layer 4
    model.add(Conv2D(filters=256, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3), kernel_size=kernelSz,
     activation='relu', padding='same' ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    # Layer 4
    model.add(Conv2D(filters=128, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3), kernel_size=kernelSz,
     activation='relu', padding='same' ))
    model.add(Dropout(0.5))

    # Layer 6
    model.add(Dense(128, activation = "relu"))
    
    # Layer 7
    model.add(Flatten())
    model.add(Dense(class_cnt, activation = "softmax"))
    
    model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    return model   



#############################################
class ShitResult:
    #67
    def get_cnn_minKernelReg44(img_h, img_w, class_cnt):
        kernelSz = (2, 2)
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(filters=32, kernel_size=kernelSz, activation='relu', input_shape = (img_h, img_w, 1), padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 2
        model.add(Conv2D(filters=128, kernel_size=kernelSz, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3), activation='relu', padding='same' ))
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 3
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=256, kernel_size=kernelSz, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3), activation='relu', padding='same' ))    
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))    
        
        # Layer 4
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=512, kernel_size=kernelSz, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3), activation='relu', padding='same' ))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))
        
        # Layer 5
        model.add(Dense(1024, activation = "relu"))
        model.add(Dropout(0.5))
        
        # Layer 6
        model.add(Dense(1024, activation = "relu"))
        model.add(Dropout(0.5))
        
        # Layer 7
        model.add(Flatten())
        model.add(Dense(class_cnt, activation = "softmax"))
        
        model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        return model


    #smaller l1 l2 values, acc ~69
    def get_cnn_minKernelReg45(img_h, img_w, class_cnt):
        kernelSz = (2, 2)
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(filters=32, kernel_size=kernelSz, activation='relu', input_shape = (img_h, img_w, 1), padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 2
        model.add(Conv2D(filters=128, kernel_size=kernelSz, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), activation='relu', padding='same' ))
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 3
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=256, kernel_size=kernelSz, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), activation='relu', padding='same' ))    
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))    
        
        # Layer 4
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=512, kernel_size=kernelSz, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), activation='relu', padding='same' ))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))
        
        # Layer 5
        model.add(Dense(1024, activation = "relu"))
        model.add(Dropout(0.5))
        
        # Layer 6
        model.add(Dense(1024, activation = "relu"))
        model.add(Dropout(0.5))
        
        # Layer 7
        model.add(Flatten())
        model.add(Dense(class_cnt, activation = "softmax"))
        
        model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        return model

    #remove second dense layer and dropout, acc 67
    def get_cnn_minKernelReg456(img_h, img_w, class_cnt):
        kernelSz = (2, 2)
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(filters=32, kernel_size=kernelSz, activation='relu', input_shape = (img_h, img_w, 1), padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 2
        model.add(Conv2D(filters=128, kernel_size=kernelSz, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), activation='relu', padding='same' ))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))    

        
        # Layer 3
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=256, kernel_size=kernelSz, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), activation='relu', padding='same' ))    
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))    
        
        # Layer 4
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=512, kernel_size=kernelSz, kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4), activation='relu', padding='same' ))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))
        
        # Layer 5
        model.add(Dense(1024, activation = "relu"))
        model.add(Dropout(0.5))
        
        
        # Layer 7
        model.add(Flatten())
        model.add(Dense(class_cnt, activation = "softmax"))
        
        model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        return model
            
    # shit, (too complex model?) loss: 0.1005 - accuracy: 0.9645 - val_loss: 2.3162 - val_accuracy: ~0.62
    def get_cnn_minKernel_noPadding(img_h, img_w, class_cnt):
        kernelSz = (2, 2)
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(filters=32, kernel_size=kernelSz, activation='relu', input_shape = (img_h, img_w, 1), padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 2
        model.add(Conv2D(filters=128, kernel_size=kernelSz, activation='relu', padding='same' ))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 3
        model.add(Conv2D(filters=256, kernel_size=kernelSz, activation='relu', padding='same' ))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))    
            
        # Layer 4
        model.add(Dense(512, activation = "relu"))
        # model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Layer 6
        model.add(Dense(256, activation = "relu"))
        # model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Layer 7
        model.add(Flatten())
        model.add(Dense(class_cnt, activation = "softmax"))
        
        model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        return model
            
    # 67 vel labai per didelis modelis
    # 144ms/step - loss: 0.1000 - accuracy: 0.9734 - val_loss: 3.8106 - val_accuracy: 0.6735
    def get_cnn_minKernelLeaky(img_h, img_w, class_cnt):
        kernelSz = (2, 2)
        alpha = 0.01 
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(filters=32, kernel_size=kernelSz, activation=LeakyReLU(alpha), input_shape = (img_h, img_w, 1), padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 2
        model.add(Conv2D(filters=128, kernel_size=kernelSz, activation=LeakyReLU(alpha), padding='same' ))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 3
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=256, kernel_size=kernelSz, activation=LeakyReLU(alpha), padding='same' ))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))    
        
        # Layer 4
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=512, kernel_size=kernelSz, activation=LeakyReLU(alpha), padding='same' ))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))
        
        # Layer 5
        model.add(Dense(1024, activation = LeakyReLU(alpha)))
        # model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Layer 6
        model.add(Dense(1024, activation = LeakyReLU(alpha)))
        # model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Layer 7
        model.add(Flatten())
        model.add(Dense(class_cnt, activation = "softmax"))
        
        model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        return model    

    #remove regulazation add back second dense layer but removed filters cnt, acc 65
    def get_cnn_minKernelReg5665(img_h, img_w, class_cnt):
        kernelSz = (2, 2)
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(filters=32, kernel_size=kernelSz, activation='relu', input_shape = (img_h, img_w, 1), padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))    

        # Layer 2
        model.add(Conv2D(filters=128, kernel_size=kernelSz, activation='relu', padding='same' ))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))    

        # Layer 3
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=256, kernel_size=kernelSz, activation='relu', padding='same' ))    
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))    
        
        # Layer 4
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=512, kernel_size=kernelSz, activation='relu', padding='same' ))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))
        
        # Layer 5
        model.add(Dense(512, activation = "relu"))
        model.add(Dropout(0.5))

        # Layer 6
        model.add(Dense(512, activation = "relu"))
        model.add(Dropout(0.5))
        
        # Layer 7
        model.add(Flatten())
        model.add(Dense(class_cnt, activation = "softmax"))
        
        model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        return model

    # NESIMOKO??? try more filters, visiskai neveikia pirmas ~30 epochu 
    def get_cnn_minKernelReg3(img_h, img_w, class_cnt):
        kernelSz = (3, 3)
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(filters=64, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3), kernel_size=(5,5),
        activation='relu', input_shape = (img_h, img_w, 1), padding='same'))
        model.add(MaxPooling2D((2, 2)))

        # Layer 2
        model.add(Conv2D(filters=128, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3), kernel_size=kernelSz,
        activation='relu', padding='same' ))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        
        # Layer 3
        model.add(Conv2D(filters=256, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3), kernel_size=kernelSz,
        activation='relu', padding='same' ))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        
        # Layer 4
        model.add(Conv2D(filters=512, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3), kernel_size=kernelSz,
        activation='relu', padding='same' ))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))

        # Layer 5
        model.add(Conv2D(filters=512, kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3), kernel_size=kernelSz,
        activation='relu', padding='same' ))
        model.add(Dropout(0.5))

        # Layer 6
        model.add(Dense(256, activation = "relu"))
        
        # Layer 7
        model.add(Flatten())
        model.add(Dense(class_cnt, activation = "softmax"))
        
        model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        return model    

    # acc ~68, (per didelis modelis)  loss: 0.0456 - accuracy: 0.9868 - val_loss: 5.1137 - val_accuracy: 0.679
    def get_cnn(img_h, img_w, class_cnt):
        # Initialize model
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape = (img_h, img_w, 1), padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 2
        model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same' ))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 3
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same' ))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))    
        
        # Layer 4
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same' ))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))
        
        # Layer 5
        model.add(Dense(1024, activation = "relu"))
        # model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Layer 6
        model.add(Dense(1024, activation = "relu"))
        # model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Layer 7
        model.add(Flatten())
        model.add(Dense(class_cnt, activation = "softmax"))
        
        model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        return model

    # acc ~69  (too complex)
    def get_cnn_minKernelElu(img_h, img_w, class_cnt):
        kernelSz = (2, 2)
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(filters=32, kernel_size=kernelSz, activation='elu', input_shape = (img_h, img_w, 1), padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 2
        model.add(Conv2D(filters=128, kernel_size=kernelSz, activation='elu', padding='same' ))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 3
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=256, kernel_size=kernelSz, activation='elu', padding='same' ))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))    
        
        # Layer 4
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=512, kernel_size=kernelSz, activation='elu', padding='same' ))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))
        
        # Layer 5
        model.add(Dense(1024, activation = "elu"))
        # model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Layer 6
        model.add(Dense(1024, activation = "elu"))
        # model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Layer 7
        model.add(Flatten())
        model.add(Dense(class_cnt, activation = "softmax"))
        
        model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        return model

    # acc 67-68 (~75 ???) vel per didelis?
    def get_cnn_minKernel(img_h, img_w, class_cnt):
        kernelSz = (2, 2)
        model = Sequential()
        
        # Layer 1
        model.add(Conv2D(filters=32, kernel_size=kernelSz, activation='relu', input_shape = (img_h, img_w, 1), padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 2
        model.add(Conv2D(filters=128, kernel_size=kernelSz, activation='relu', padding='same' ))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        
        # Layer 3
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=256, kernel_size=kernelSz, activation='relu', padding='same' ))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.1))    
        
        # Layer 4
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters=512, kernel_size=kernelSz, activation='relu', padding='same' ))
        # model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.5))
        
        # Layer 5
        model.add(Dense(1024, activation = "relu"))
        # model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Layer 6
        model.add(Dense(1024, activation = "relu"))
        # model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Layer 7
        model.add(Flatten())
        model.add(Dense(class_cnt, activation = "softmax"))
        
        model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        return model








