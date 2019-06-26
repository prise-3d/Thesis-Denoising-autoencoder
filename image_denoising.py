from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

from modules.utils import config as cfg
import cv2

import argparse

def generate_model(input_shape=(3, 200, 200)):
    print(input_shape)
    input_img = Input(shape=input_shape)  # adapt this if using `channels_first` image data format

    x = Conv3D(32, (1, 3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling3D((1, 2, 2), padding='same')(x)
    x = Conv3D(32, (1, 3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling3D((1, 2, 2), padding='same')(x)


    x = Conv3D(32, (1, 3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling3D((1, 2, 2))(x)
    x = Conv3D(32, (1, 3, 3), activation='relu', padding='same')(x)
    x = UpSampling3D((1, 2, 2))(x)
    decoded = Conv3D(1, (1, 3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)

    # TODO : check if 
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder

def main():

    # load params
    parser = argparse.ArgumentParser(description="Train Keras model and save it into .json file")

    parser.add_argument('--data', type=str, help='dataset filename prefix (without .train and .test)', required=True)
    parser.add_argument('--output', type=str, help='output file name desired for model (without .json extension)', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size used as model input', default=cfg.keras_batch)
    parser.add_argument('--epochs', type=int, help='number of epochs used for training model', default=cfg.keras_epochs)
   
    args = parser.parse_args()

    p_data_file  = args.data
    p_output     = args.output
    p_batch_size = args.batch_size
    p_epochs     = args.epochs

    # load data from `p_data_file`
    ########################
    # 1. Get and prepare data
    ########################
    print("Preparing data...")
    dataset_train = pd.read_csv(p_data_file + '.train', header=None, sep=";")
    dataset_test = pd.read_csv(p_data_file + '.test', header=None, sep=";")

    print("Train set size : ", len(dataset_train))
    print("Test set size : ", len(dataset_test))

    # default first shuffle of data
    dataset_train = shuffle(dataset_train)
    dataset_test = shuffle(dataset_test)

    print("Reading all images data...")

    # getting number of chanel
    n_channels = len(dataset_train[1][1].split('::'))
    print("Number of channels : ", n_channels)

    img_width, img_height = cfg.keras_img_size

    # specify the number of dimensions
    if K.image_data_format() == 'channels_first':
        if n_channels > 1:
            input_shape = (1, n_channels, img_width, img_height)
        else:
            input_shape = (n_channels, img_width, img_height)

    else:
        if n_channels > 1:
            input_shape = (1, img_width, img_height, n_channels)
        else:
            input_shape = (img_width, img_height, n_channels)

    # `:` is the separator used for getting each img path
    if n_channels > 1:
        dataset_train[1] = dataset_train[1].apply(lambda x: [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in x.split('::')])
        dataset_test[1] = dataset_test[1].apply(lambda x: [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in x.split('::')])
    else:
        dataset_train[1] = dataset_train[1].apply(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE))
        dataset_test[1] = dataset_test[1].apply(lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE))

    x_dataset_train = dataset_train[1].apply(lambda x: np.array(x).reshape(input_shape))
    x_dataset_test = dataset_test[1].apply(lambda x: np.array(x).reshape(input_shape))
    
    y_dataset_train = dataset_train[0].apply(lambda x: cv2.imread(x).reshape(input_shape))
    y_dataset_test = dataset_test[0].apply(lambda x: cv2.imread(x).reshape(input_shape))

    # format correct data
    x_data_train = np.array([item for item in x_dataset_train.values])
    #x_data_train = np.array(x_dataset_train.values)
    x_data_test = np.array([item for item in x_dataset_test.values])
    #x_data_test = np.array(x_dataset_test.values)

    y_data_train = np.array([item for item in y_dataset_train.values])
    #y_data_train = np.array(y_dataset_train.values)
    y_data_test = np.array([item for item in y_dataset_test.values])
    #y_data_test = np.array(y_dataset_test.values)

    # load model
    autoencoder = generate_model(input_shape)

    # tensorboard --logdir=/tmp/autoencoder
    autoencoder.fit(x_data_train, y_data_train,
                    epochs=100,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(x_data_test, y_data_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=0, write_graph=False)])

    # save model
    
if __name__ == "__main__":
    main()