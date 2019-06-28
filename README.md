# Denoising with autoencoder

## Description

Utilisation d'un autoencoder pour apprendre statistiquement comment il est possible de générer une image de synthèse.

Input : 
- Noisy image
- Z-buffer
- Normal card

or other information...

Output :
- Reference image

## Requirements

```bash
git clone --recursive https://github.com/prise-3d/Thesis-Denoising-autoencoder.git XXXXX
```

```bash
pip install -r requirements.txt
```

## How to use ?

[Autoencoder keras documentation](https://blog.keras.io/building-autoencoders-in-keras.html)

Generate reconstructed data from specific method of reconstruction (run only once time or clean data folder before):
```
python generate/generate_reconstructed_data.py -h
```

Generate custom dataset from one reconstructed method or multiples (implemented later)
```
python generate/generate_dataset.py -h
```

### Reconstruction parameter (--params)

List of expected parameter by reconstruction method:
- **svd_reconstruction:** Singular Values Decomposition
  - Param definition: *interval data used for reconstruction (begin, end)*
  - Example: *"100, 200"*
- **ipca_reconstruction:** Iterative Principal Component Analysis
  - Param definition: *number of components used for compression and batch size*
  - Example: *"30, 35"*
- **fast_ica_reconstruction:**  Fast Iterative Component Analysis
  - Param definition: *number of components used for compression*
  - Example: *"50"*
- **static** Use static file to manage (such as z-buffer, normals card...)
  - Param definition: *Name of image of scene need to be in {sceneName}/static/xxxx.png*
  - Example: *"img.png"*

**__Example:__**
```bash
python generate/generate_dataset.py --output data/output_data_filename --metrics "svd_reconstruction, ipca_reconstruction, fast_ica_reconstruction" --renderer "maxwell" --scenes "A, D, G, H" --params "100, 200 :: 50, 10 :: 50" --nb_zones 10 --random 1 --only_noisy 1
```

Then, run the model:
```bash
python image_denoising --data data/my_dataset --output output_model_name
```

## License

[The MIT license](https://github.com/prise-3d/Thesis-NoiseDetection-metrics/blob/master/LICENSE)
