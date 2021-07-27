import numpy as np
#from matplotlib import pyplot as plt
import tifffile

img_array = np.load('C:/Users/user/Desktop/test/result.npy')
#print(img_array)
#plt.imshow(img_array)

tifffile.imwrite('C:/Users/user/Desktop/test/temp.tif', img_array, photometric='minisblack')
#tifffile.imwrite('C:/Users/user/Desktop/test/temp.tif', img_array, photometric='rgb')