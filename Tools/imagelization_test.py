'''
Convert the matrix into a visual image form
'''

import numpy as np
#from matplotlib import pyplot as plt
import tifffile
import os

file_path = "/home/lihangjun/dataset/npy/"
path_list = os.listdir(file_path) 

path_name = []
count = 0

for i in path_list:
    path_name.append(i)
    print(i)

path_name.sort()


def tifflization(path, num):
    img_array = np.load(file_path+path)
    #print(img_array)
    #plt.imshow(img_array)

    if "F" in path:
        tifffile.imwrite('/home/lihangjun/dataset/tiff/'+str(num)+".tif", img_array, photometric='minisblack')
        #tifffile.imwrite('/home/lihangjun/dataset/tiff/'+str(num)+".tif", img_array, photometric='rgb')
    else:
        tifffile.imwrite('/home/lihangjun/dataset/tiff/'+str(num)+"_F.tif", img_array, photometric='minisblack')

#tifflization
for file_name in path_name:
    count = count+1
    tifflization(file_name, count)
    print(count)