class waterMarkEmbedding(object):
    def __init__(self):
        pass
    
    def convert8x8Group(self,ori_mat,method = 'divide'):
        # Divide Grayscale Image or 2-D matrix to 8x8 block or Merge 8x8 block to 2-D matrix 
        # some row below or colunm right may be dropped if rows or columns mod 8 != 0
        # matrix will be divided after cut to 8n x 8m or list be merged to a martix

        import numpy as np
        from PIL import Image

        assert ((method == 'divide' and isinstance(ori_mat,(np.ndarray,Image.Image))) or 
                (method == 'merge' and isinstance(ori_mat,(tuple,list)))),\
            TypeError('Only accept PIL.Image,np.ndarray when method divide and list or tuple when merge')

        if isinstance(ori_mat,(np.ndarray,Image.Image)) and len(np.array(ori_mat).shape) != 2:
            raise ValueError('Matrix input has wrong dim, only accept grayscale image.(dim = 2)')

        if method == 'divide':
            ori_mat = np.array(ori_mat)
            ori_mat = ori_mat[:ori_mat.shape[0] - ori_mat.shape[0] % 8,:ori_mat.shape[1] - ori_mat.shape[1] % 8]
            return [np.hsplit(col_mat,col_mat.shape[1] / 8) for col_mat in np.vsplit(np.array(ori_mat),np.array(ori_mat).shape[0] / 8)]
        elif method == 'merge':
            return np.vstack([np.hstack(mat) for mat in ori_mat]).astype(np.uint8)     
        
    def DCTwaterMarkEmbedding4Gray(self,carrier,key_seq,watermark = None,method = 'enc',watermark_shape = None):
        # 采用论文所述中频系数的方法，对中频系数和量化增量要求高，需要调参，同时不确定是否对图片有依赖性
        # 不进行上述操作，则会出现严重的失真，对于混沌加密的图像造成的失真更为严重。
        # parameter carrier   : grayscale image,larger than watermark.size * 16
        # parameter watermark : grayscale image
        
        import numpy as np
        from cv2 import dct,idct
        import pdb
        
        assert method in ('enc','dec'),ValueError('Parameter \'method\' only accept {}'.format(('enc','dec')))

        carrier = np.array(carrier)
        assert len(carrier.shape) == 2 ,TypeError('parameter carrier only accept PIL.Image,np.ndarray of grayscale') 
        carrier_dct = [[dct(col_mat.astype(np.float64)) for col_mat in row_mat] for row_mat in self.convert8x8Group(carrier)]
        
        ent_list = list(np.array(key_seq) % 8)
        ent_list = sorted(list(set(ent_list)),key = ent_list.index)
        
        if method == 'enc':
            watermark = np.array(watermark)
            assert len(watermark.shape) == 2,TypeError('parameter watermark only accept PIL.Image,np.ndarray of grayscale')

            assert (np.array(carrier.shape) >= np.array(watermark.shape) * 16).all(),'the carrier is not big enough.'
            
            for row in range(watermark.shape[0]):
                for col in range(watermark.shape[1]):
                    dct_increment = (np.array(list(bin(watermark[row,col])[2:].rjust(8,'0'))).astype(np.float64) - 0.5) * 4
                    blocks_dctmean = np.mean((carrier_dct[2 * row + 1][2 * col    ],
                                              carrier_dct[2 * row    ][2 * col + 1],
                                              carrier_dct[2 * row + 1][2 * col + 1]),axis = 0)
                    
                    carrier_dct[2 * row][2 * col][range(8),ent_list] = blocks_dctmean[range(8),ent_list] + dct_increment
                                                                       
            return self.convert8x8Group([[idct(col_mat) for col_mat in row_mat] for row_mat in carrier_dct],'merge'),watermark.shape
        elif method == 'dec':
            assert isinstance(watermark_shape,(list,tuple)) and len(watermark_shape) == 2,\
                TypeError('Parameter \'watermark_shape\' only accept {} of length 2'.format(('list','tuple')))
            watermark = np.zeros(watermark_shape,dtype = np.uint8)
            for row in range(watermark_shape[0]):
                for col in range(watermark_shape[1]):
                    blocks_dctmean = np.mean((carrier_dct[2 * row + 1][2 * col    ],
                                              carrier_dct[2 * row    ][2 * col + 1],
                                              carrier_dct[2 * row + 1][2 * col + 1]),axis = 0)
                    try:
                        watermark[row,col] = int(((carrier_dct[2 * row][2 * col][range(8),ent_list] > blocks_dctmean[range(8),ent_list]).astype(np.int8) + ord('0')).tobytes().decode('ascii'),2)
                    except:
                        import pdb
                        pdb.set_trace()
            return watermark,watermark.shape
  
    def DCTwaterMarkEmbedding4RGB(self,carrier,key_seq,watermark = None,method = 'enc',watermark_shape = None):
        # parameter carrier   : RGB image,larger than watermark.size * 16
        # parameter watermark : RGB image
        
        import numpy as np
        from PIL import Image
        from ImageProcess import smColorLayer
        
        assert method in ('enc','dec'),ValueError('Parameter \'method\' only accept {}'.format(('enc','dec')))

        carrier = np.array(carrier)
        assert len(carrier.shape) == 3 and (carrier.shape[-1] == 3 or carrier.shape[0] == 3),\
            TypeError('parameter carrier only accept PIL.Image,np.ndarray of RGB or split RGB colorLayer Group')
        
        if carrier.shape[-1] == 3:
            carrier = smColorLayer(carrier,'split')
            
        if watermark is not None:
            watermark = np.array(watermark)
            assert len(watermark.shape) == 3 and (watermark.shape[-1] == 3 or watermark.shape[0] == 3),\
                TypeError('parameter watermark only accept PIL.Image,np.ndarray of RGB or split RGB colorLayer Group')
            if watermark.shape[-1] == 3:
                watermark = smColorLayer(watermark,'split')            
            
        if watermark_shape != None and len(watermark_shape) == 3:
            watermark_shape = (watermark_shape[1:] if watermark_shape[0] == 3 else watermark_shape[:-1])
        
        return_c = [0,0,0]
        
        if method == 'enc':
            for w_c,c_c,i in zip(watermark,carrier,range(3)):
                return_c[i],watermark_shape = self.DCTwaterMarkEmbedding4Gray(c_c,key_seq,w_c,'enc')
            
        elif method == 'dec':
            for c_c,i in zip(carrier,range(3)):
                return_c[i],watermark_shape = self.DCTwaterMarkEmbedding4Gray(c_c,key_seq,method = 'dec',watermark_shape = watermark_shape)
        
        return smColorLayer(return_c,'merge'),watermark_shape
    