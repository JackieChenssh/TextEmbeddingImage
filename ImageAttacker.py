from enum import Enum

class FilterType(Enum):
    Median   = 1
    Maximum  = 2
    Minimum  = 3
    Gaussian = 4
    Smooth   = 5
    Shapen   = 6
    Mean     = 7
    
class NoiseType(Enum):
    Gaussian = 8
    Uniform  = 9    

class ImageAttacker(object):
    def __init__(self):
        self.paddingDict = {       FilterType.Gaussian   : [[ 1, 2, 1],[ 2, 3, 2],[ 1, 2, 1]],
                             FilterType.Smooth    : [[ 1, 2, 1],[ 2, 4, 2],[ 1, 2, 1]],
                             FilterType.Shapen    : [[-1,-1,-1],[-1, 9,-1],[-1,-1,-1]],
                             FilterType.Mean     : [[ 1, 1, 1],[ 1, 1, 1],[ 1, 1, 1]]}
        
    def convolution(self,mat,padding):
        #return (C,H,W) or (H,W)
        import numpy as np
        from math import floor
        from tqdm.notebook import trange
        
        mat,padding = np.array(mat),np.array(padding)

        padding = padding / np.sum(padding)
        
        assert padding.shape[0] == padding.shape[1],ValueError('Padding must be (a group of) Square Matrix.')
        assert len(padding.shape) == len(mat.shape),ValueError('Mat and Padding must have the same depth.')
        assert padding.shape[0] > 2,ValueError('Padding\'s size should be larger than 2.')
        
        mat = np.vstack((mat[0 : floor(padding.shape[0] / 2)],mat,mat[-floor(padding.shape[0] / 2) - 1 : -1])).T
        mat = np.vstack((mat[0 : floor(padding.shape[0] / 2)],mat,mat[-floor(padding.shape[0] / 2) - 1 : -1])).T

        return np.array([[np.sum(mat[row : row + padding.shape[0],col : col + padding.shape[0]] * padding) 
                          for col in range(mat.shape[1] - padding.shape[0] + 1)] for row in trange(mat.shape[0] - padding.shape[0] + 1)]).astype(np.uint8)
    
    def AdditionNoise(self,mat,method = NoiseType.Gaussian):
        import numpy as np
        
        mat = np.asarray(mat)
        
        if method == NoiseType.Gaussian:
            return (mat + np.random.random(mat.shape) * 5).astype(np.uint8)
        elif method == NoiseType.Uniform:
            return (mat + np.random.uniform(0, np.std(mat) / 10, mat.shape)).astype(np.uint8)
    
    def ImageFilter(self,mat,padding = None, method = None):
        from PIL import Image
        from tqdm.notebook import trange
        from math import floor
        import numpy as np
        
        mat = np.asarray(mat)
        
        if not padding:
            paddingShape = (3,3)
            
            mat = np.vstack((mat[0 : floor(paddingShape[0] / 2)],mat,mat[-floor(paddingShape[0] / 2) - 1 : -1])).T
            mat = np.vstack((mat[0 : floor(paddingShape[0] / 2)],mat,mat[-floor(paddingShape[0] / 2) - 1 : -1])).T
            
            if method == FilterType.Median:
                return np.array([[np.median(mat[row : row + paddingShape[0],col : col + paddingShape[0]]) 
                                  for col in range(mat.shape[1] - paddingShape[0] + 1)] for row in trange(mat.shape[0] - paddingShape[0] + 1)],dtype = np.uint8)
            elif method == FilterType.Maximum:
                return np.array([[np.max(mat[row : row + paddingShape[0],col : col + paddingShape[0]]) 
                                  for col in range(mat.shape[1] - paddingShape[0] + 1)] for row in trange(mat.shape[0] - paddingShape[0] + 1)],dtype = np.uint8)
            elif method == FilterType.Minimum:
                return np.array([[np.min(mat[row : row + paddingShape[0],col : col + paddingShape[0]]) 
                                  for col in range(mat.shape[1] - paddingShape[0] + 1)] for row in trange(mat.shape[0] - paddingShape[0] + 1)],dtype = np.uint8)
            elif self.paddingDict.get(method):
                return self.convolution(mat,self.paddingDict[method])
            else:
                raise Exception()
        else: 
            return self.convolution(mat,padding)