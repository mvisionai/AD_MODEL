from keras.layers import Conv3D, MaxPool3D, Flatten, Dense,MaxPooling3D,UpSampling3D,merge,concatenate,Activation
from keras.layers.convolutional import Convolution3D
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.losses import categorical_crossentropy
from  keras import  optimizers
from matplotlib.pyplot import cm
from keras.models import Sequential, Model
import  tensorflow as tf
import numpy as np
from AD_Dataset import  Dataset_Import
import keras
from  numpy import  random
import h5py
import  tensorflow as tf

import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=""


def unet_model(nb_filter=32, dim=80, clen=2, img_rows=100,
                             img_cols=80):  # NOTE that this procedure is/should be used with img_rows & img_cols as None

    # aiming for architecture similar to the http://cs231n.stanford.edu/reports2016/317_Report.pdf
    # Our model is six layers deep, consisting  of  a  series  of  three  CONV-RELU-POOL  layyers (with 32, 32, and 64 3x3 filters), a CONV-RELU layer (with 128 3x3 filters), three UPSCALE-CONV-RELU lay- ers (with 64, 32, and 32 3x3 filters), and a final 1x1 CONV- SIGMOID layer to output pixel-level predictions. Its struc- ture resembles Figure 2, though with the number of pixels, filters, and levels as described here

    ## 3D CNN version of a previously developed unet_model_xd_6j
    zconv = clen
    model = Sequential()
    inputs = Input(( dim, img_rows, img_cols,1))

    model.add(Conv3D(filters=nb_filter, kernel_size=(2,2,2),activation='relu',padding='same'))
    model.add(Conv3D(filters=nb_filter, kernel_size=zconv, strides=clen,activation='relu',padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2),padding="same"))




    model.add(Conv3D(filters=2 * nb_filter,kernel_size= zconv, activation='relu', padding='same'))
    model.add(Conv3D(filters=2 * nb_filter,kernel_size= zconv, activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(filters=4 * nb_filter,kernel_size= zconv, activation='relu', padding='same'))
    model.add(Conv3D(filters=4 * nb_filter,kernel_size= zconv, activation='relu', padding='same'))

    model.add(Conv3D(filters=2 * nb_filter,kernel_size= zconv, activation='relu', padding='same'))
    model.add(Conv3D(filters=2 * nb_filter,kernel_size= zconv, activation='relu', padding='same'))

    model.add(Conv3D(filters=nb_filter, kernel_size=zconv, activation='relu', padding='same'))
    model.add(Conv3D(filters=nb_filter, kernel_size=zconv, activation='relu', padding='same'))

    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(filters=2 * nb_filter, kernel_size=zconv,strides= 1, activation='relu', padding='same'))
    model.add(Conv3D(filters=2 * nb_filter, kernel_size=zconv,strides= 1, activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(filters=2 * nb_filter, kernel_size=zconv,strides= 1, activation='relu', padding='same'))
    model.add(Conv3D(filters=2 * nb_filter, kernel_size=zconv,strides= 1, activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

  # need one extra layer to get to 1D x 2D mask ...
    model.add(Conv3D(filters=2 * nb_filter, kernel_size=zconv,strides= 1 ,activation='relu', padding='same'))
    model.add(Conv3D(filters=2 * nb_filter, kernel_size=zconv,strides= 1, activation='relu', padding='same'))
    model.add(MaxPool3D(pool_size=(1, 1, 1)))

    model.add(Flatten())
    model.add(Dense(units=1500, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(units=2, activation='softmax'))

    datafeed = Dataset_Import()
    source_data = datafeed.all_source_data(augment_data=datafeed.data_augmentation)
    validation_datas = datafeed.all_target_data()

    print(len(source_data))

    datafeed.set_random_seed(random.random_integers(1000))
    source_training_data = datafeed.shuffle(source_data)

    source_feed = datafeed.next_batch_combined(len(source_training_data), source_training_data)
    data_source = [data for data in source_feed]

    # training data
    data_source = np.asarray(data_source)
    y_train = keras.utils.to_categorical(data_source[0:, 1], 2)

    # validation_data
    validation_datas = datafeed.shuffle(validation_datas)
    validate_feed = datafeed.next_batch_combined(len(validation_datas), validation_datas)
    validate_source = [data for data in validate_feed]
    validate_source = np.asarray(validate_source)
    y_validate = keras.utils.to_categorical(validate_source[0:, 1], 2)

    ## define the model with input layer and output layer

    sgd = optimizers.SGD(lr=0.001, decay=0.01, momentum=0.90, nesterov=False)

    model.compile(optimizer=sgd, loss=categorical_crossentropy, metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),  loss=dice_coef_loss, metrics=[dice_coef])

    model.fit(x=np.array([data for data in (data_source[:, 0])]), y=y_train, batch_size=5, epochs=100,
              validation_data=(np.array([data for data in (validate_source[:, 0])]), y_validate))
    model.save_weights("best_weights.h5")
    #return  model

def alextNet():  ## input layer

    model = Sequential()
    # input_layer = Input((79,95,79,1))

    ## convolutional layers
    model.add(Conv3D(filters=64, kernel_size=(9, 9, 9), strides=4,padding="same", input_shape=(80, 100, 80, 1)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))


    model.add(Conv3D(filters=128, kernel_size=(5, 5, 5),padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))
    print(model.get_output_shape_at(0))

    model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), activation='relu' ,padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(filters=512, kernel_size=(3, 3, 3),padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2)))

    model.add(Conv3D(filters=1024, kernel_size=(3, 3, 3),padding="same"))
    #model.add(MaxPool3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))


    model.add(Flatten())

    model.add(Dense(units=4096, activation='relu'))
    model.add(Dense(units=1000, activation='relu'))
    #model.add(Dense(units=1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=2, activation='softmax'))

    datafeed = Dataset_Import()
    source_data = datafeed.all_source_data(augment_data=datafeed.data_augmentation)
    validation_datas = datafeed.all_target_data()

    print(len(source_data))

    datafeed.set_random_seed(random.random_integers(1000))
    source_training_data = datafeed.shuffle(source_data)

    source_feed = datafeed.next_batch_combined(len(source_training_data), source_training_data)
    data_source = [data for data in source_feed]

    # training data
    data_source = np.asarray(data_source)
    y_train = keras.utils.to_categorical(data_source[0:, 1], 2)

    # validation_data
    validation_datas = datafeed.shuffle(validation_datas)
    validate_feed = datafeed.next_batch_combined(len(validation_datas), validation_datas)
    validate_source = [data for data in validate_feed]
    validate_source = np.asarray(validate_source)
    y_validate = keras.utils.to_categorical(validate_source[0:, 1], 2)

    ## define the model with input layer and output layer
    # model = Model(inputs=input_layer, outputs=output_layer)1e-6

    sgd = optimizers.SGD(lr=0.001, decay=0.01, momentum=0.93, nesterov=True)
    model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
    model.fit(x=np.array([data for data in (data_source[:, 0])]), y=y_train, batch_size=5, epochs=300,
              validation_data=(np.array([data for data in (validate_source[:, 0])]), y_validate))
    model.save_weights("best_weights.h5")


if __name__== "__main__":

    alextNet()
    #unet_model()




