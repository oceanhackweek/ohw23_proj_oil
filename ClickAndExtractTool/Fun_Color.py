import numpy as np
import cv2
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors

#from PIL import Image
#im2 = cv2.imread('img/IlhabelaSP_20210509_123508.jpg')
#im3 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
#im4 = cv2.imread("path/to/file/image.jpg")[:,:,::-1]
# HSI --> Computer vision

# Constants: # OpenCV is BGR, PIL is RGB, mpl is RGB
H=0; S=1; I=2
R=0; G=1; B=2


# RGB TO CMY
# https://code.adonline.id.au/cmyk-in-python/
def rgb_to_cmy(img, img_color_mode="BGR"):
    if img_color_mode =="BGR":
        B=0; G=1; R=2
    else:
        R=0; G=1; B=2
        
    img_ = img.astype(float)/255.

    # Extract channels and convert the input BGR image to CMYK colorspace
    with np.errstate(invalid='ignore', divide='ignore'):
        K = 1 - np.max(img_, axis=2)
        C = (((1-img_[..., R] - K)/(1-K)) * 255).astype(np.uint8)
        M = (((1-img_[..., G] - K)/(1-K)) * 255).astype(np.uint8)
        Y = (((1-img_[..., B] - K)/(1-K)) * 255).astype(np.uint8)

    return (C,M,Y)


# Retorna
# . RGB quando avg(RGB) > lim_I
# . (0,0,0) otherwise
def filterI(img, lim_I):
    img2 = np.apply_along_axis(np.avg, 2, img)


# img = image to be grey scaled
# by = method of grey scalling: AVG | STD | 
# Default channel in BGR dispose (OpenCV default)
# There are multiple implementations of the grayscale conversion in play. cvtColor() is THE opencv implementation and will be consistent across platforms. 
# By using imread() you are at the mercy of the platform-specific implementation of imread().
def greyScaleBy(img, by="AVG", img_color_mode="BGR"):
    if img_color_mode =="BGR":
        B=0; G=1; R=2
    else:
        R=0; G=1; B=2
        
    ret2D = None
    
    # Standard deviation of RGB channels
    if by == "STD": 
       ret2D = np.std(img, axis=2)
       
    # The Average method, simply average the values: ( R + G + B)/3
    if by == "AVG":
       ret2D = np.average(img, axis=2)
       
    # There are many formulas for the Luminance, depending on the R,G,B color primaries:
    # The weighted average method, colors are not weighted equally. Since pure green is lighter than pure red and pure blue, 
    # it has a higher weight. Pure blue is the darkest of the three, so it receives the least weight.
    # Used by cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) and other softwares
    # REC 601
    # https://github.com/opencv/opencv/blob/master/modules/imgproc/src/color.simd_helpers.hpp#L22
    # constants for conversion from/to RGB and Gray, YUV, YCrCb according to BT.601
    # static const float B2YF = 0.114f;
    # static const float G2YF = 0.587f;
    # static const float R2YF = 0.299f;

    if by == "WEIGHTED": # IDEM cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       ret2D = img[:, :, R]*0.299 + img[:, :, G]*0.587 + img[:, :, B]*0.114
        
    # The Luminosity method 0.21R + 0.71G + 0.07B. For human perception. 
    # Humans are more sensitive to green than other colors, so green is weighted most heavily
    # The Luminosity method tends to reduce contrast. It works best overall and it
    # is the default method used in most applications. Some images look better
    # using one of the other algorithms. Sometimes the four methods produce
    # very similar result. Rec.709/EBU:  Y = 0.213*R + 0.715*G + 0.072*B
    # REC 709 (luminance)
    if by == "HDTV" or by == "HUMAN": 
       ret2D = img[:, :, R]*0.2126 + img[:, :, G]*0.7152 + img[:, :, B]*0.0722

    if by == "LUMINANCE2": 
       ret2D = np.sqrt(np.square(img[:, :, R]*0.2126) + np.square(img[:, :, G]*0.7152) + np.square(img[:, :, B]*0.0722))

    # REC 2100
    if by == "HDR": 
       ret2D = img[:, :, R]*0.2627 + img[:, :, G]*0.6780 + img[:, :, B]*0.0593
       
    # The Lightness method, averages the most prominent and least prominent colors ( max (R,G,B) + min (R,G,B) ) / 2
    if by == "LIGHTNESS": 
       ret2D = np.round((np.max(img, axis=2) + np.min(img, axis=2)) / 2)

    # Dynamic range (max diff)
    # Diff between the higher vs. the lower RGB values
    if by == "MAX_DIFF": 
       ret2D = (np.max(img, axis=2) - np.min(img, axis=2))
    
    ret2D = ret2D.astype(np.uint8) 
        
    return(ret2D)

# The Gaussian Filter is a low pass filter. The Gaussian smoothing (or blur) of an image removes the outlier pixels or the 
# high-frequency components to reduce noise. It is likewise utilized as a preprocessing stage prior to applying our AI or deep learning models.
# . 
# . ksize    - Gaussian kernel size (width and height), the width and height can have different values and must be positive and odd,
# . sigma_x  - Gaussian kernel standard deviation along X-axis,
# . dst      - output image,
# . sigma_y  - Gaussian kernel standard deviation along Y-axis
# . border_type - image boundaries. Possible values are - BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101, BORDER_TRANSPARENT, BORDER_REFLECT101, BORDER_DEFAULT, BORDER_ISOLATED
# Sigma especifies how wide the curve should be inside the kernel. If only sigma_x is specified, sigma_y is automatically taken as equal to sigma_x. 
# If both are defined as zeros, they are calculated from the kernel size.
def blurImage(img):
    return(cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT))


# pal=["black", "grey", "white", "blue", "darkblue"]; n=256
def getColors(pal, n):
    # Color palettes
    my_cmap = LinearSegmentedColormap.from_list("my_cmap", pal, N=n)
    my_hex = [colors.rgb2hex(my_cmap(i)[:3]) for i in range(256)]
    rgb_list = [list(np.array(colors.to_rgb(h))) for h in my_hex]
    rgb_list = [[round(rgb[0] * 255.), round(rgb[1] * 255.), round(rgb[2] * 255.)]  for rgb in rgb_list]
    return(rgb_list)


# Map the colors using a lookup table: 
# In OpenCV you can apply a colormap stored in a 256 x 1 color image to an image using a lookup table LUT.
def rgb2LkUpTb(rgb_list):
    if np.max(rgb_list) == 1: # rgb in 0-1 range
       multi_factor = 255.
    else:
       multi_factor = 1
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    lut[:, 0, :] = [[round(rgb[0] * multi_factor), round(rgb[1] * multi_factor), round(rgb[2] * multi_factor)] for rgb in rgb_list]
    return(lut)
    
    
    
# img = image to be analysed
# channel = int 0:...  *** CUIDADO: IMAGEM PODE ESTAR EM BGR ou RGB
# fudge_factor = aguns calculos precisam que não sejam feitos sobre "0", adiciona o valor aos canais
# Precisa retornar ima imagem grey scaled (com R=G=B) para aplicar colormap
# channel="R"; fudge_factor=0.1; xmode="RGB"
def getImgChannelPower(img, channel, fudge_factor=0.1, xmode="BGR"):
    if xmode =="BGR":
        dic_channel = {"B":0, "G":1, "R":2}
    else:
        dic_channel = {"R":0, "G":1, "B":2}
    img = np.int32(img)  # fazer contas pode dar overflow com uint8
    other_channels = list(set([0,1,2]) - set([dic_channel[channel]]))
    res = (img[:,:, dic_channel[channel]] + fudge_factor) / ((img[:,:, other_channels[0]] + img[:,:, other_channels[1]] + (fudge_factor*2))/2)
    return(res)
    
    #print(res.min(), res.max())
    
    # pal=["black", "grey", "white", "blue", "darkblue"]
    # my_cmap = LinearSegmentedColormap.from_list("my_cmap", pal, N=256)
    # norm = colors.Normalize(np.min(res), np.max(res))
    
    # plt.imshow(res)
    # plt.colorbar()    
    
    # ret = np.zeros(res.shape + (3,))
    # ret[:, :, 0] = res.ravel()

    # 1080*1440

    # res = np.full_like(res, 127, dtype=np.uint8)
    # ret = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    
    # len(res+res+res)
        
    # ret = np.reshape([res]*3, res.shape + (3,))
    
    # 1080*1440*3
    
    # np.concatenate((res, res, res), axis=1)
    # return((img[:,:, channel] + fudge_factor) / (((img[:,:, other_channels[0]] + img[:,:, other_channels[1]])/2) + fudge_factor))

#https://pyimagesearch.com/2015/09/28/implementing-the-max-rgb-filter-in-opencv/
# Canal maior mantem valor e outros zeram, se empate no maior mantém e zera-se o menor
def max_rgb(image):
	# split the image into its BGR components
	(B, G, R) = cv2.split(image)
	# find the maximum pixel intensity values for each
	# (x, y)-coordinate,, then set all pixel values less
	# than M to zero
	M = np.maximum(np.maximum(R, G), B)
	R[R < M] = 0
	G[G < M] = 0
	B[B < M] = 0
	# merge the channels back together and return the image
	return cv2.merge([B, G, R])

# Canal maior vai pro máximo valor (255) e outros zeram, se empate no maior os dois para o maximo e zera-se o menor
#image=img
def boost_rgb(image):
	# split the image into its BGR components
	(B, G, R) = cv2.split(image)
	# find the maximum pixel intensity values for each
	# (x, y)-coordinate,, then set all pixel values less
	# than M to zero
	M = np.maximum(np.maximum(R, G), B)
	R[R >= M] = 255
	G[G >= M] = 255
	B[B >= M] = 255
	R[R < M] = 0
	G[G < M] = 0
	B[B < M] = 0
	# merge the channels back together and return the image
	return cv2.merge([B, G, R])


# Define modo de cores de acordo com o tipo da imagem   
def getColorMode(img):
   G=1
   if type(img) == np.ndarray:
       R=2; B=0  # OpenCV
   else:
       R=0; B=2  # PIL

   return (R, G, B)


# https://chart-studio.plotly.com/~empet/15229/heatmap-with-a-discrete-colorscale/#/
def discrete_colorscale(bvals, colors):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors)+1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)     
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values
    
    dcolorscale = [] #discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
    return dcolorscale    

# x is a tuple (R,G,B)
def rgb_to_hsi(x, xmode="RGB"):  # # OpenCV is BGR, PIL is RGB, mpl is RGB
    #print(x[R])
    if xmode == "RGB":
        R=0; G=1; B=2
    else:    # BGR
        B=0; G=1; R=2
        
    r = float(x[R])
    g = float(x[G])
    b = float(x[B])
    high = max(r, g, b)
    mean = sum((r, g, b))/3
    low = min(r, g, b)
    h, s, i = high, high, mean

    d = high - low
    s = 0 if high == 0 else d/high

    if high == low:
        h = 0.0
    else:
        h = {
            r: (g - b) / d + (6 if g < b else 0),
            g: (b - r) / d + 2,
            b: (r - g) / d + 4,
        }[high]
        h /= 6

    return h, s, i
#rgb_to_hsi(im2[0,0,0], im2[0,0,1], im2[0,0,2])

# Analyses the tuple RGB
# Cloud regions in color aerial photographs usually have higher intensity and lower hue. For an input color aerial photograph, we first
# transform it from RGB color model into HSI color model, and then we construct a significance map to highlight the difference between 
# cloud regions and non-cloud regions as follows W = Intensity + "E" / Ihue + "E", We bound the intensity and hue to [0, 1] to compute 
# the significance map, which is proved to be a better significance map. "E" is an amplification factor, in our paper, we typically set
# "E" = 1.0.
# --
# Luminance = 0.3r + 0.59g + 0.11b (Pratt W (2007) Digital image processing. Wiley-Interscience.)
# GIMP and rgb2gray() Matlab function
# --
# Luminance = .2126r + .7152g + 0.0722b 
# int() if because of the overflow

#x=(172, 175, 179); xmode="BGR"
def rgbExpl(x, xmode="RGB"):  # # OpenCV is BGR, PIL is RGB, mpl is RGB
   #print(x)
   if xmode == "RGB":
      R=0; G=1; B=2
   else:    # BGR
      B=0; G=1; R=2

   E = 1 # fudge factor 
   r = 2 # rounding digit
   hsi  = rgb_to_hsi(x, xmode)        # returned "i" is my old brightness (just the mean values of RGB)
   sd   = np.round(np.std(x), r)
   var  = np.round(sd ** 2, r)
   lum  = np.round( (0.2126*x[R]) + (.7152*x[G]) + (0.0722*x[B]), r)    # REF
   w    = np.round( (hsi[I]+E) / (hsi[H]+E), r)                         # REF
   w2   = np.round( (hsi[I]+E) / (hsi[S]+E), r)                         # ZR
   powR = np.round((int(x[R])+E) / (((int(x[G]) + int(x[B]))/2)+E), r)  # ZR
   powG = np.round((int(x[G])+E) / (((int(x[R]) + int(x[B]))/2)+E), r)  # ZR
   powB = np.round((int(x[B])+E) / (((int(x[R]) + int(x[G]))/2)+E), r)  # ZR
   avg = np.mean(x)  # the mean is the I of HSI
   # "H","S","I","SD","VAR","LUM","W","W2","R_POW","G_POW","B_POW"
   return[round(hsi[H], r), round(hsi[S], r), round(hsi[I], r), sd, var, lum, w, w2, powR, powG, powB, np.round(avg, r)]
#rgbExpl(x)

rgbExplVec = lambda x: rgbExpl(x)  # Vectorized version
