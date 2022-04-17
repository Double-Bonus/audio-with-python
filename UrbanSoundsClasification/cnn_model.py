from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, ELU




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

# 
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












#####################################################

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