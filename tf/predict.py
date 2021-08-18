import tensorflow as tf
from cnn import AlexNet
import numpy as np
import os
import random


model = AlexNet().get_model()
model.load_weights("8_epoch.hdf5")

def predict(img1,img2):
    image1 = tf.io.read_file(img1)
    image2 = tf.io.read_file(img2)
    
    image1 = tf.image.decode_jpeg(image1, channels=3)
    image2 = tf.image.decode_jpeg(image2, channels=3)
    # resize as 224，224，3
    image1 = tf.image.resize(image1, (224, 224))
    image2 = tf.image.resize(image2, (224, 224))
    # Combine two images according to the channel
    image_combined = tf.concat([image1, image2], 2)
   
    predicted = model.predict(tf.reshape(image_combined, [1, 224, 224, 6]), steps=1)
   #print(predicted)
    predicted=predicted[0]
    label_ind = np.argmax(predicted)
    
    return int(label_ind)


pathdir = os.listdir("/mnt/train_data/compare/VeCom_0330-0430_1/")

for i in range(99):
    samples = random.sample(pathdir, 2)
    print(predict("/mnt/train_data/compare/VeCom_0330-0430_1/"+samples[0],"/mnt/train_data/compare/VeCom_0330-0430_1/"+samples[1]))