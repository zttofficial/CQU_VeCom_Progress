import numpy as np
from PIL import Image


def multichannels_splicing(path1 ,path2, count):
    img1,img2 = Image.open(path1), Image.open(path2)
    img1 = img1.resize((1440, 1080), Image.ANTIALIAS)
    img2 = img2.resize((1440, 1080), Image.ANTIALIAS)
    data1 = np.asarray(img1)
    data2 = np.asarray(img2)
    #check shape
    #print(data1.shape)
    #print(data2.shape)
    result = np.concatenate((data1, data2), axis=2)
    #print(result.shape)
    np.save('D:/CQU/车身对比/npy/'+str(count), result)
    
    #service for tifffile.py
    return(result, data1, data2)
