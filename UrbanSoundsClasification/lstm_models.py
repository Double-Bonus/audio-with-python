from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D
from keras.layers import TimeDistributed, LSTM, ConvLSTM2D    

from keras import regularizers
    

def get_lstm(x_train, class_cnt, modelType):
    lstm = Sequential()
    cnn = Sequential()

    if modelType == 0:  # acc 59, train time ~ 0:45 min
        lstm.add(ConvLSTM2D(10, kernel_size=5, strides=(1, 1), activation='elu', padding='same', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(MaxPooling2D(pool_size=(2, 2)))
        lstm.add(Dropout(0.25))
        lstm.add(Conv2D(16, kernel_size=4, strides=(1, 1), activation='elu', padding='same'))
        lstm.add(MaxPooling2D(pool_size=(2, 2)))
        lstm.add(Dropout(0.25))
        lstm.add(Conv2D(32, kernel_size=3, strides=(1, 1), activation='elu', padding='same'))
        lstm.add(MaxPooling2D(pool_size=(2, 2)))
        lstm.add(Dropout(0.25))
        lstm.add(Flatten())
        lstm.add(Dense(32, activation='elu'))

    elif modelType == 1: # acc 59 
        lstm.add(ConvLSTM2D(8, kernel_size=(3, 3), strides=(1, 1), activation='elu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(MaxPooling2D(pool_size=(2, 2)))
        lstm.add(Dropout(0.25))
        lstm.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        lstm.add(MaxPooling2D(pool_size=(2, 1)))
        lstm.add(Dropout(0.25))
        lstm.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        lstm.add(MaxPooling2D(pool_size=(2, 1)))
        lstm.add(Dropout(0.25))
        lstm.add(Flatten())
        lstm.add(Dense(32, activation='elu'))

    elif modelType == 2: # acc ~57+, train time 1:00min epoch
        cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(MaxPooling2D(pool_size=(1, 1)))
        cnn.add(Dropout(0.25))

        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(ConvLSTM2D(16, kernel_size=(3, 3), strides=(1, 1), dropout=0.25, recurrent_dropout=0.25))
        lstm.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        lstm.add(MaxPooling2D(pool_size=(1, 2)))
        lstm.add(Dropout(0.25))
        lstm.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        lstm.add(MaxPooling2D(pool_size=(2, 2)))
        lstm.add(Dropout(0.25))
        lstm.add(Flatten())

    elif modelType == 3: # acc  ~69 train time - 5s
        cnn.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='elu', input_shape=(x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=(3, 2)))
        cnn.add(Dropout(0.25))

        cnn.add(ZeroPadding2D((1, 2)))
        cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Dropout(0.25))

        cnn.add(ZeroPadding2D((1, 2)))
        cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Dropout(0.25))
        cnn.add(Flatten())

        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, dropout=0.0, recurrent_dropout=0.0))

    elif modelType == 4: # acc ~67, train time 4s
        cnn.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='elu', input_shape=(x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=(2, 3)))
        cnn.add(Dropout(0.25))        
        cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Dropout(0.25))
        cnn.add(Flatten())

        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, dropout=0.0, recurrent_dropout=0.0))

    elif modelType == 5: # acc ~ 62, train time 4s
        cnn.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='elu', input_shape=(x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=(3, 2)))
        cnn.add(Dropout(0.1))
        cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=(2, 1)))
        cnn.add(Dropout(0.1))
        cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=(2, 1)))
        cnn.add(Dropout(0.5))
        cnn.add(Flatten())

        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, dropout=0.0, recurrent_dropout=0.0))

    elif modelType == 6: # 69 acc

        # Layer 1
        cnn.add(Conv2D(filters=36, kernel_size=5, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(MaxPooling2D((3, 2),strides=3 ))

        # Layer 2
        cnn.add(Conv2D(filters=48, kernel_size=2, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
        cnn.add(MaxPooling2D((2, 1), strides=2 ))

        # Layer 3
        cnn.add(Conv2D(filters=60, kernel_size=2, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))    
        cnn.add(GlobalAveragePooling2D())
        
        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, dropout=0.0, recurrent_dropout=0.0))
    

    elif modelType == 7: # 70 acc

        # Layer 1
        cnn.add(Conv2D(filters=36, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(MaxPooling2D((3, 2),strides=3 ))

        # Layer 2
        cnn.add(Conv2D(filters=48, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='same' ))
        cnn.add(MaxPooling2D((2, 1), strides=2 ))

        # Layer 3
        cnn.add(Conv2D(filters=60, kernel_size=2, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='same' ))    
        cnn.add(GlobalAveragePooling2D())
        
        # # Layer 4
        # cnn.add(Dense(72, activation = "relu"))
        # cnn.add(Dropout(0.5))


        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, dropout=0.0, recurrent_dropout=0.0))


    elif modelType == 8: # 71.5 acc  

        # Layer 1
        cnn.add(Conv2D(filters=36, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(MaxPooling2D((3, 2),strides=3 ))

        # Layer 2
        cnn.add(ZeroPadding2D((1, 2)))
        cnn.add(Conv2D(filters=48, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
        cnn.add(MaxPooling2D((2, 1), strides=2 ))

        # Layer 3
        cnn.add(Conv2D(filters=60, kernel_size=2, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))    
        cnn.add(GlobalAveragePooling2D())
        
        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, dropout=0.0, recurrent_dropout=0.0))

    elif modelType == 9: # 62 acc 
        # paslkutiniam LSTM lsuoksni filteriu daugiau bandoma
        # Layer 1
        cnn.add(Conv2D(filters=36, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(MaxPooling2D((3, 2),strides=3 ))

        # Layer 2
        cnn.add(ZeroPadding2D((1, 2)))
        cnn.add(Conv2D(filters=48, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
        cnn.add(MaxPooling2D((2, 1), strides=2 ))

        # Layer 3
        cnn.add(Conv2D(filters=60, kernel_size=2, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))    
        cnn.add(GlobalAveragePooling2D())
        
        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(64, dropout=0.0, recurrent_dropout=0.0))


    elif modelType == 10: # 70 acc
        #dropout ir recurent proout, ilgai mokosi bet lyg progreso yra?
        # Layer 1
        cnn.add(Conv2D(filters=36, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(MaxPooling2D((3, 2),strides=3 ))

        # Layer 2
        cnn.add(ZeroPadding2D((1, 2)))
        cnn.add(Conv2D(filters=48, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
        cnn.add(MaxPooling2D((2, 1), strides=2 ))

        # Layer 3
        cnn.add(Conv2D(filters=60, kernel_size=2, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))    
        cnn.add(GlobalAveragePooling2D())
        
        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, dropout=0.25, recurrent_dropout=0.1))

    elif modelType == 11: # ~55 acc,  relu i LSTM layer activation
        # Layer 1
        cnn.add(Conv2D(filters=36, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(MaxPooling2D((3, 2),strides=3 ))

        # Layer 2
        cnn.add(ZeroPadding2D((1, 2)))
        cnn.add(Conv2D(filters=48, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
        cnn.add(MaxPooling2D((2, 1), strides=2 ))

        # Layer 3
        cnn.add(Conv2D(filters=60, kernel_size=2, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))    
        cnn.add(GlobalAveragePooling2D())
        
        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, activation='relu', dropout=0.0, recurrent_dropout=0.0))

    elif modelType == 12: # 72.5 acc,  more filters, added dropout in covn2
        # Dropout is good
        # Layer 1
        cnn.add(Conv2D(filters=64, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(MaxPooling2D((3, 2),strides=3 ))
        cnn.add(Dropout(0.25))


        # Layer 2
        cnn.add(ZeroPadding2D((1, 2)))
        cnn.add(Conv2D(filters=88, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
        cnn.add(MaxPooling2D((2, 1), strides=2 ))
        cnn.add(Dropout(0.25))


        # Layer 3
        cnn.add(Conv2D(filters=128, kernel_size=2, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))    
        cnn.add(GlobalAveragePooling2D())
        
        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, dropout=0.0, recurrent_dropout=0.0))

    elif modelType == 13: # 75 acc
        # recurent drop 0.2 duoda geresni rezultata
        # Layer 1
        cnn.add(Conv2D(filters=64, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', 
            input_shape = (x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(MaxPooling2D((3, 2),strides=3 ))
        cnn.add(Dropout(0.25))

        # Layer 2
        cnn.add(ZeroPadding2D((1, 2)))
        cnn.add(Conv2D(filters=88, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
        cnn.add(MaxPooling2D((2, 1), strides=2 ))
        cnn.add(Dropout(0.25))

        # Layer 3
        cnn.add(Conv2D(filters=128, kernel_size=2, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))    
        cnn.add(GlobalAveragePooling2D())
        
        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, dropout=0.0, recurrent_dropout=0.2)) # makes trainig way longer

    elif modelType == 14: # 74.3 acc
        # recurent drop bandm 0.4, nepadidejo zenklaiai
        # Layer 1
        cnn.add(Conv2D(filters=64, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(MaxPooling2D((3, 2),strides=3 ))
        cnn.add(Dropout(0.25))


        # Layer 2
        cnn.add(ZeroPadding2D((1, 2)))
        cnn.add(Conv2D(filters=88, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
        cnn.add(MaxPooling2D((2, 1), strides=2 ))
        cnn.add(Dropout(0.25))


        # Layer 3
        cnn.add(Conv2D(filters=128, kernel_size=2, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))    
        cnn.add(GlobalAveragePooling2D())
        
        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, dropout=0.0, recurrent_dropout=0.4))

    elif modelType == 15: # 71.8  acc
        # add dropout to recurent droptot
        # Layer 1
        cnn.add(Conv2D(filters=64, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(MaxPooling2D((3, 2),strides=3 ))
        # cnn.add(Dropout(0.25))


        # Layer 2
        cnn.add(ZeroPadding2D((1, 2)))
        cnn.add(Conv2D(filters=88, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
        cnn.add(MaxPooling2D((2, 1), strides=2 ))
        # cnn.add(Dropout(0.25))


        # Layer 3
        cnn.add(Conv2D(filters=128, kernel_size=2, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))    
        cnn.add(GlobalAveragePooling2D())
        
        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, dropout=0.25, recurrent_dropout=0.2))

    elif modelType == 16: #  acc
        # 13 model but larger
        # Layer 1
        cnn.add(Conv2D(filters=96, kernel_size=4, kernel_regularizer=regularizers.l2(1e-3), activation='relu', input_shape = (x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(MaxPooling2D((3, 2),strides=3 ))
        cnn.add(Dropout(0.25))

        # Layer 2
        cnn.add(ZeroPadding2D((1, 2)))
        cnn.add(Conv2D(filters=124, kernel_size=3, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))
        cnn.add(MaxPooling2D((2, 1), strides=2 ))
        cnn.add(Dropout(0.25))

        # Layer 3
        cnn.add(Conv2D(filters=256, kernel_size=2, kernel_regularizer=regularizers.l2(1e-3), activation='relu', padding='valid' ))    
        cnn.add(GlobalAveragePooling2D())
        
        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, dropout=0.0, recurrent_dropout=0.2))
    lstm.add(Dense(class_cnt, activation='softmax')) # all models uses same output
    return cnn, lstm