import numpy as np
from PIL import Image

path1 = "D:/CQU/车身对比/车身比对0506-0623_1/3200000100_21042432000601250526_0111_0_.jpg"
path2 = "D:/CQU/车身对比/车身比对0506-0623_1/3200000100_21042432000601250526_0111_0_A.jpg"

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
np.save('C:/Users/user/Desktop/New folder/result', result)
