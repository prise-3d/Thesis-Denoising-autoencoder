from modules.config.cnn_config import *

# store all variables from cnn config
context_vars = vars()

# Custom config used for redefined config variables if necessary

# folders

## noisy_folder                    = 'noisy'
## not_noisy_folder                = 'notNoisy'

# file or extensions

## post_image_name_separator       = '___'

# variables

## features_choices_labels         = ['static', 'svd_reconstruction', 'fast_ica_reconstruction', 'ipca_reconstruction']

# parameters

keras_epochs                    = 200
## keras_batch                     = 32
## val_dataset_size                = 0.2

## keras_img_size                  = (200, 200)