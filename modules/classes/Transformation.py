import os

from transformation_functions import svd_reconstruction, fast_ica_reconstruction, ipca_reconstruction

# Transformation class to store transformation method of image and get usefull information
class Transformation():

    def __init__(self, _transformation, _param):
        self.transformation = _transformation
        self.param = _param

    def getTransformedImage(self, img):

        if self.transformation == 'svd_reconstruction':
            begin, end = list(map(int, self.param.split(',')))
            data = svd_reconstruction(img, [begin, end])

        if self.transformation == 'ipca_reconstruction':
            n_components, batch_size = list(map(int, self.param.split(',')))
            data = ipca_reconstruction(img, n_components, batch_size)

        if self.transformation == 'fast_ica_reconstruction':
            n_components = self.param
            data = fast_ica_reconstruction(img, n_components)

        return data
    
    def getTransformationPath(self):

        path = self.transformation

        if self.transformation == 'svd_reconstruction':
            begin, end = list(map(int, self.param.split(',')))
            path = os.path.join(path, str(begin) + '_' + str(end))

        if self.transformation == 'ipca_reconstruction':
            n_components, batch_size = list(map(int, self.param.split(',')))
            path = os.path.join(path, 'N' + str(n_components) + '_' + str(batch_size))

        if self.transformation == 'fast_ica_reconstruction':
            n_components = self.param
            path = os.path.join(path, 'N' + str(n_components))

        return path

    def getName(self):
        return self.transformation

    def getParam(self):
        return self.param

    def __str__( self ):
        return self.transformation + ' transformation with parameter : ' + self.param