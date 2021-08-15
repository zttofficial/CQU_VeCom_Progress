import argparse
import csv
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tqdm import tqdm
import utils
from cnn import AlexNet

from tf_veri_dataset import make_tf_veri_dataset


#path = './0111-1130-A'
path = '/mnt/train_data/compare/VeCom_0330-0430_1'
#path = '/mnt/train_data/compare/VeCom_0330-0430_1'
img_list = os.listdir(path)
test_img_list = []
for i in img_list:
    if 'A' not in i:
        test_img_list.append(i.split(".")[0])

# print(test_img_list)
model = AlexNet().get_model()
model.summary()
#model.load_weights("./models/alex_net_weights_last_05141549.h5")
model.load_weights("./models/latest.h5")
def test(img1,img2):
    image1 = tf.io.read_file(img1)
    image2 = tf.io.read_file(img2)
    #
    image1 = tf.image.decode_jpeg(image1, channels=3)
    image2 = tf.image.decode_jpeg(image2, channels=3)
    # resize图像为224，224，3
    image1 = tf.image.resize(image1, (224, 224))
    image2 = tf.image.resize(image2, (224, 224))
    # 按照通道合并两张图像
    image_combined = tf.concat([image1, image2], 2)
   

    predicted = model.predict(tf.reshape(image_combined, [1, 224, 224, 6]), steps=1)
   #print(predicted)
    predicted=predicted[0]
    label_ind = np.argmax(predicted)
    
    return int(label_ind)








total = 0
TP = 0
TS = 1
FS = 1
FN = 0

# f = open("./train/test.txt","r",encoding="utf-8").readlines()

# for i in tqdm(range(len(f))):
#     file1,file2,label = f[i].strip().split(",")
    
#     if int(label)==1 and os.path.exists(file1) and os.path.exists(file2):
#         TS =TS +1
#         if test(file1,file2) == 1:
#             TP = TP +1


    
#     if  int(label)==0 and os.path.exists(file1) and os.path.exists(file2):
#         FS = FS + 1
#         if test(file1,file2) == 0:
#            FN = FN +1
#     if i %100==0:
#         print("TP/TS", float(TP)/TS, TP, TS)
#         print("FN/FS", float(FN)/FS, FN, FS)
# print("TP/TS", float(TP)/TS, TP, TS)
# print("FN/FS", float(FN)/FS, FN, FS)

for i in tqdm(range(len(test_img_list))):
    name = test_img_list[i]
 
    if os.path.exists(path+"/"+name+".jpg") and os.path.exists(path+"/"+ name+"A.jpg"):
        TS =TS +1
        if test(path+"/"+ name+"A.jpg",path+"/"+name+".jpg") == 1:
            TP = TP +1


    ran = random.choice(test_img_list)
    #print(ran)
    if os.path.exists(path+"/"+name+".jpg") and os.path.exists(os.path.join(path,ran.split(".")[0]+"A.jpg")):
        FS = FS + 1
        if test(os.path.join(path,ran.split(".")[0]+"A.jpg"),path+"/"+name+".jpg") == 0:
           FN = FN +1
    if i %100==0:
        print("TP/TS", float(TP)/TS, TP, TS)
        print("FN/FS", float(FN)/FS, FN, FS)
print("TP/TS", float(TP)/TS, TP, TS)
print("FN/FS", float(FN)/FS, FN, FS)





'''
print(test("./img/1.jpg","./img/2.jpg"))
print(test("./img/4.jpg","./img/6.jpg"))
'''


#imglist = ['./img/0001_c001_00016460_0.jpg', './img/0002_c002_00030625_1.jpg']


