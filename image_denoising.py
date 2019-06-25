from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard

import argparse

def generate_model(input_shape=(3, 200, 200)):
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
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return autoencoder

def main():

    # load params
    parser = argparse.ArgumentParser(description="Train Keras model and save it into .json file")

    parser.add_argument('--data', type=str, help='dataset filename prefix (without .train and .test)', required=True)
    parser.add_argument('--output', type=str, help='output file name desired for model (without .json extension)', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size used as model input')
    parser.add_argument('--epochs', type=int, help='number of epochs used for training model')
   
    args = parser.parse_args()

    p_data_file  = args.data
    p_output     = args.output
    p_batch_size = args.batch_size
    p_epochs     = args.epochs

    # load data from `p_data_file`
    x_train_noisy = []
    x_train       = []
    x_test_noisy  = []
    x_test        = []

    # load model
    autoencoder = generate_model()

    # tensorboard --logdir=/tmp/autoencoder
    autoencoder.fit(x_train_noisy, x_train,
                    epochs=100,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=0, write_graph=False)])

    # save model
    
if __name__ == "__main__":
    main()