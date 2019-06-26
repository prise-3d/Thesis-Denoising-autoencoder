from numpy.linalg import svd
from sklearn.decomposition import FastICA, IncrementalPCA

import numpy as np

from ipfml import metrics

def svd_reconstruction(img, interval):
    
    begin, end = interval
    lab_img = metrics.get_LAB_L(img)
    lab_img = np.array(lab_img, 'uint8')
    
    U, s, V = svd(lab_img, full_matrices=True)
    
    # reconstruction using specific interval
    smat = np.zeros((end-begin, end-begin), dtype=complex)
    smat[:, :] = np.diag(s[begin:end])
    output_img = np.dot(U[:, begin:end],  np.dot(smat, V[begin:end, :]))
        
    return output_img


def fast_ica_reconstruction(img, components):

    lab_img = metrics.get_LAB_L(img)
    lab_img = np.array(lab_img, 'uint8')

    ica = FastICA(n_components = 50)
    # run ICA on image
    ica.fit(lab_img)
    # reconstruct image with independent components
    image_ica = ica.fit_transform(lab_img)
    restored_image = ica.inverse_transform(image_ica)

    return restored_image


def ipca_reconstruction(img, components, _batch_size=25):

    lab_img = metrics.get_LAB_L(img)
    lab_img = np.array(lab_img, 'uint8')

    transformer = IncrementalPCA(n_components=components, batch_size=_batch_size)

    transformed_image = transformer.fit_transform(lab_img) 
    restored_image = transformer.inverse_transform(transformed_image)

    return restored_image