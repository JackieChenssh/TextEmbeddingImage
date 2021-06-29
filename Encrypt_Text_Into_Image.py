from RandomCorEnc4Text import RandomCorEnc4Text
from waterMarkEmbedding import waterMarkEmbedding
from ImageProcess import imageScrambling
from ImageAttacker import ImageAttacker,FilterType,NoiseType

from PIL import Image
import numpy as np

text_key = 'happy_sugar_life'
test_text  = 'You must - there are over 200,000 words in our free online dictionary, but you are looking for one that\'s only in the Merriam-Webster Unabridged Dictionary.'
watermark_img = Image.open('./img/img_watermark.jpg')
carrier_img = Image.open('./img/img_carrier.jpg')
carrier_img = carrier_img.resize(np.array(carrier_img.size) * 2)

md5_key,watermarked_img = RandomCorEnc4Text().convertEncAndDec(text_key,test_text,watermark_img,'enc')

logic_ini = np.var(md5_key) - np.var(md5_key,dtype = int)
scrambled_img = imageScrambling(watermarked_img,logic_ini)

encoded_img,watermark_shape = waterMarkEmbedding().DCTwaterMarkEmbedding4RGB(carrier_img, md5_key, scrambled_img)

encoded_img.save('./img/img_encoded_src.bmp')

Image.merge('RGB',[Image.fromarray(ImageAttacker().ImageFilter(colorLayer,method = FilterType.Median)) for colorLayer in encoded_img.split()]).save('./img/img_encoded_Median.bmp')

Image.merge('RGB',[Image.fromarray(ImageAttacker().ImageFilter(colorLayer,method = FilterType.Maximum)) for colorLayer in encoded_img.split()]).save('./img/img_encoded_Maximum.bmp')

Image.merge('RGB',[Image.fromarray(ImageAttacker().ImageFilter(colorLayer,method = FilterType.Minimum)) for colorLayer in encoded_img.split()]).save('./img/img_encoded_Minimum.bmp')

Image.merge('RGB',[Image.fromarray(ImageAttacker().ImageFilter(colorLayer,method = FilterType.Gaussian)) for colorLayer in encoded_img.split()]).save('./img/img_encoded_Gaussian.bmp')

Image.merge('RGB',[Image.fromarray(ImageAttacker().ImageFilter(colorLayer,method = FilterType.Smooth)) for colorLayer in encoded_img.split()]).save('./img/img_encoded_Smooth.bmp')

Image.merge('RGB',[Image.fromarray(ImageAttacker().ImageFilter(colorLayer,method = FilterType.Shapen)) for colorLayer in encoded_img.split()]).save('./img/img_encoded_Shapen.bmp')

Image.merge('RGB',[Image.fromarray(ImageAttacker().ImageFilter(colorLayer,method = FilterType.Mean)) for colorLayer in encoded_img.split()]).save('./img/img_encoded_Mean.bmp')

Image.merge('RGB',[Image.fromarray(ImageAttacker().AdditionNoise(colorLayer,method = NoiseType.Gaussian)) for colorLayer in encoded_img.split()]).save('./img/img_encoded_GaussianNoise.bmp')

Image.merge('RGB',[Image.fromarray(ImageAttacker().AdditionNoise(colorLayer,method = NoiseType.Uniform)) for colorLayer in encoded_img.split()]).save('./img/img_encoded_UniformNoise.bmp')

encoded_img = Image.open('./img/img_encoded_src.bmp')
decoded_img,_ = waterMarkEmbedding().DCTwaterMarkEmbedding4RGB(encoded_img, md5_key,method = 'dec',watermark_shape = watermark_shape)
watered_img = imageScrambling(decoded_img,logic_ini,'dec')
RandomCorEnc4Text().convertEncAndDec(text_key,carrier_img = watered_img,method = 'dec')

encoded_img = Image.open('./img/img_encoded_Median.bmp')
decoded_img,_ = waterMarkEmbedding().DCTwaterMarkEmbedding4RGB(encoded_img, md5_key,method = 'dec',watermark_shape = watermark_shape)
watered_img = imageScrambling(decoded_img,logic_ini,'dec')
RandomCorEnc4Text().convertEncAndDec(text_key,carrier_img = watered_img,method = 'dec')

encoded_img = Image.open('./img/img_encoded_Minimum.bmp')
decoded_img,_ = waterMarkEmbedding().DCTwaterMarkEmbedding4RGB(encoded_img, md5_key,method = 'dec',watermark_shape = watermark_shape)
watered_img = imageScrambling(decoded_img,logic_ini,'dec')
RandomCorEnc4Text().convertEncAndDec(text_key,carrier_img = watered_img,method = 'dec')

encoded_img = Image.open('./img/img_encoded_Maximum.bmp')
decoded_img,_ = waterMarkEmbedding().DCTwaterMarkEmbedding4RGB(encoded_img, md5_key,method = 'dec',watermark_shape = watermark_shape)
watered_img = imageScrambling(decoded_img,logic_ini,'dec')
RandomCorEnc4Text().convertEncAndDec(text_key,carrier_img = watered_img,method = 'dec')

encoded_img = Image.open('./img/img_encoded_Gaussian.bmp')
decoded_img,_ = waterMarkEmbedding().DCTwaterMarkEmbedding4RGB(encoded_img, md5_key,method = 'dec',watermark_shape = watermark_shape)
watered_img = imageScrambling(decoded_img,logic_ini,'dec')
RandomCorEnc4Text().convertEncAndDec(text_key,carrier_img = watered_img,method = 'dec')

encoded_img = Image.open('./img/img_encoded_Smooth.bmp')
decoded_img,_ = waterMarkEmbedding().DCTwaterMarkEmbedding4RGB(encoded_img, md5_key,method = 'dec',watermark_shape = watermark_shape)
watered_img = imageScrambling(decoded_img,logic_ini,'dec')
RandomCorEnc4Text().convertEncAndDec(text_key,carrier_img = watered_img,method = 'dec')

encoded_img = Image.open('./img/img_encoded_Shapen.bmp')
decoded_img,_ = waterMarkEmbedding().DCTwaterMarkEmbedding4RGB(encoded_img, md5_key,method = 'dec',watermark_shape = watermark_shape)
watered_img = imageScrambling(decoded_img,logic_ini,'dec')
RandomCorEnc4Text().convertEncAndDec(text_key,carrier_img = watered_img,method = 'dec')

encoded_img = Image.open('./img/img_encoded_Mean.bmp')
decoded_img,_ = waterMarkEmbedding().DCTwaterMarkEmbedding4RGB(encoded_img, md5_key,method = 'dec',watermark_shape = watermark_shape)
watered_img = imageScrambling(decoded_img,logic_ini,'dec')
RandomCorEnc4Text().convertEncAndDec(text_key,carrier_img = watered_img,method = 'dec')

encoded_img = Image.open('./img/img_encoded_GaussianNoise.bmp')
decoded_img,_ = waterMarkEmbedding().DCTwaterMarkEmbedding4RGB(encoded_img, md5_key,method = 'dec',watermark_shape = watermark_shape)
watered_img = imageScrambling(decoded_img,logic_ini,'dec')
RandomCorEnc4Text().convertEncAndDec(text_key,carrier_img = watered_img,method = 'dec')

encoded_img = Image.open('./img/img_encoded_UniformNoise.bmp')
decoded_img,_ = waterMarkEmbedding().DCTwaterMarkEmbedding4RGB(encoded_img, md5_key,method = 'dec',watermark_shape = watermark_shape)
watered_img = imageScrambling(decoded_img,logic_ini,'dec')
RandomCorEnc4Text().convertEncAndDec(text_key,carrier_img = watered_img,method = 'dec')