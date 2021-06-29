def smColorLayer(colorLayers,method):
    import numpy as np
    from PIL import Image
    import pdb
    
    assert method in ('merge','split'),ValueError('Parameter \'method\' only accept {}'.format(('merge','split')))

    img = None
    
    if method == 'merge':
        img = Image.merge('RGB',[Image.fromarray(colorLayer) for colorLayer in colorLayers])
    elif method == 'split':
        img = np.array([np.array(colorLayer) for colorLayer in Image.fromarray(np.asarray(colorLayers)).split()])
    return img

def swapColorLayerRGBBGR(imgArray):
    # Convert RGB To BGR or BGR to RGB
    import numpy as np
    from PIL import Image

    transformed = False
    if not isinstance(imgArray,(np.ndarray,Image.Image)):
        raise TypeError('Input must be numpy.ndarray or Image.Image type.')

    if isinstance(imgArray,Image.Image):
        imgArray = np.array(imgArray)
        transformed = True

    if imgArray.shape[-1] == 3:
        if len(imgArray.shape) == 4:
            imgArray[:,:,:,[0,2]] = imgArray[:,:,:,[2,0]]        
        elif len(imgArray.shape) == 3:
            imgArray[:,:,[0,2]] = imgArray[:,:,[2,0]] 
        else:
            raise ValueError('Input must be matrix (Channel,Height,Width) or (SampleID,Channel,Height,Width)')
    else:
        raise ValueError('Input must be RGB Image or Image_Group')

    if transformed:
        imgArray = Image.fromarray(imgArray)

    return imgArray

def imageScrambling(ori_mat,logic_ini,method = 'enc'):
    # Image encryption and scrambling algorithm based on chaotic sequence
    # return a mat of encrypted or decrypted img
    import numpy as np
    from PIL import Image
    from tqdm import trange

    assert isinstance(logic_ini,float) and abs(logic_ini) > 0 and logic_ini < 1,\
        ValueError('Parameter \'logic_ini\' only accept number in (0,1)')
    assert isinstance(ori_mat,(list,tuple,np.ndarray,Image.Image)), \
            TypeError('Only accept PIL.Image,np.ndarray,list or tuple')
    assert method in ('enc','dec'),ValueError('Parameter \'method\' only accept {}'.format(('enc','dec')))

    ori_type = type(ori_mat)
    ori_mat = np.array(ori_mat)
    ori_shape = ori_mat.shape
    ori_mat = ori_mat.reshape(-1)

    logic_seq = [logic_ini] * len(ori_mat)

    # Calculate Logictic Sequence
    varmu = 3.57
    for i in range(1,len(logic_seq)):
        logic_seq[i] = varmu * logic_seq[i - 1] * (1.0 - logic_seq[i - 1])

#     varlambda = 2
#     for i in trange(1,len(logic_seq)):
#         logic_seq[i] = 1.0 - varlambda * np.pow(logic_seq[i - 1],2)


    if method == 'enc':
        return Image.fromarray(ori_mat[np.argsort(logic_seq)].reshape(ori_shape))
    elif method == 'dec':
        dec_mat = np.zeros(len(logic_seq),dtype = np.uint8)
        dec_mat[np.argsort(logic_seq)] = ori_mat  
        return Image.fromarray(dec_mat.reshape(ori_shape))