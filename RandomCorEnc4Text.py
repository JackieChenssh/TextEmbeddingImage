class RandomCorEnc4Text(object):
    # Hide text into image,RGB or Grayscale
    def __init__(self):
        return
    
    def getCorList(self,src_key,img_shape = (1,1,1)):
        assert isinstance(src_key,str),ValueError('src_key only accept str')
        import numpy as np
        import hashlib

        code_list = {'0':0 , '1':1 , '2':2 , '3':3 , '4':4 , '5':5 , '6':6 ,
                     '7':7 , '8':8 , '9':9 , 'a':0xa,'b':0xb,'c':0xc,'d':0xd,'e':0xe,'f':0xf}

        md5_int = [code_list[word] for word in hashlib.md5(src_key.encode('utf8')).hexdigest()]
        
        cor_list = {tuple(np.prod(tuple(md5_int[m] for m in [i,j,k][:len(img_shape)])) % np.array(img_shape)) : None
                  for i in range(len(md5_int)) for j in range(i,len(md5_int)) for k in range(j,len(md5_int))}

        if cor_list.get(tuple(0 for i in range(len(img_shape)))):
            del cor_list[tuple(0 for i in range(len(img_shape)))]

        return np.array(md5_int),np.array(list(cor_list.keys()))
    
    def convertEncAndDec(self,src_key,encoding_str = None,carrier_img = None,method = 'enc'):
        # parameter src_key        : type str, source_key used to encrypt str
        # parameter encoding_str   : type str, str ready to be encrypted,no need for method == 'dec'
        # parameter carrier_img    : type PIL.Image or numpy.ndarray, to contain encoded str  
        # parameter method         : enum ('enc','dec'),'enc' for encrypt and 'dec' for decrypt
        # return (method == 'enc') : ent_key(type list,element int),carrier_img(type with encrypted str in)
        # return (method == 'dec') : a string equal to encoding_str, the length equal to min(source_str.length,cor_list.maxlength)

        from PIL import Image
        import numpy as np
        from tqdm import trange,tqdm_notebook
        
        assert (src_key == None or isinstance(src_key,str)) and isinstance(src_key,str),\
                ValueError('encoding_str or src_key only accept str')
        assert isinstance(carrier_img,(Image.Image,np.ndarray)),TypeError('carrier_img only accept PIL.Image,np.ndarray')
        assert method in ('enc','dec'),ValueError('Parameter \'method\' only accept {}'.format(('enc','dec')))
        
        carrier_img = np.array(carrier_img)
        ent_key,cor_list = self.getCorList(src_key,carrier_img.shape)
        if method == 'enc':
            encoding_asc = [ord(c) for c in encoding_str][:len(cor_list)]
            cor_list = cor_list[:len(encoding_asc)]

            carrier_img[0,0] = ([(len(encoding_asc) >> 16) % 256,(len(encoding_asc) >> 8) % 256, len(encoding_asc) % 256] if len(carrier_img.shape) == 3 
                                else len(encoding_asc) % 256)
            
            carrier_img[tuple(np.vsplit(cor_list.T,len(carrier_img.shape)))] = encoding_asc
            
            return ent_key,Image.fromarray(carrier_img)
        
        elif method == 'dec':
            dec_str = carrier_img[tuple(np.vsplit(cor_list[:np.sum([carrier_img[0,0,i] << 8 * i for i in range(3)]) if len(carrier_img.shape) == 3 
                        else carrier_img[0,0]].T,len(carrier_img.shape)))][0]
            dec_str = dec_str[dec_str >= ord(' ')]
            dec_str = dec_str[dec_str <= ord('~')]
            return dec_str.tobytes().decode("ascii")
#             return dec_str