# @Author: WL
# @Date  :  2020/12/06
# coding:utf-8

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D

class AlexNet(Layer):
    '''
    Alex Net
    '''
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = Sequential()
        # 卷积层1
        # 卷积层Conv2D(卷积核数目,(卷积核长，卷积核宽) strides=(x,y步长) input_shape，padding, activation))
        self.model.add(Conv2D(filters=96, kernel_size=11, strides=4, input_shape=(224, 224, 6), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=3, strides=2))
        # 卷积层2
        self.model.add(Conv2D(256, kernel_size=5, padding='same', activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=3, strides=2))
        # 卷积层3,4,5
        self.model.add(Conv2D(384, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Conv2D(384, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=3, strides=2))
        # 全连接层6，7，8
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        # 模型编译
        # self.model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizers.Adam(0.01), metrics=['accuracy'])

    def get_model(self):
        self.model.summary()
        return self.model


class VGGNet_11(Layer):
    '''
    Alex Net
    '''
    def __init__(self):
        super(VGGNet_11, self).__init__()
        self.model = Sequential()
        # 卷积层1
        # 卷积层Conv2D(卷积核数目,(卷积核长，卷积核宽) strides=(x,y步长) input_shape，padding, activation))
        # padding='same':表示如果卷积核扫描结束后还剩1个元素，不够卷积核扫描，则在后面补0
        # padding='valid':表示将不足以卷积核扫描的元素抛弃
        self.model.add(Conv2D(filters=64, kernel_size=3, input_shape=(224, 224, 6), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=2, strides=2))
        # 卷积层2
        self.model.add(Conv2D(128, kernel_size=3, padding='same', activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=2, strides=2))
        # 卷积层3,4
        self.model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2, strides=2))
        # 卷积层5,6
        self.model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=2, strides=2))
        # 卷积层7,8
        self.model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=2, strides=2))
        # 全连接层9,10,11
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))

    def get_model(self):
        self.model.summary()
        return self.model


class ResNet(Layer):
    '''
    Alex Net
    '''
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = Sequential()
        # 卷积层1
        self.model.add(Conv2D(filters=64, kernel_size=7, input_shape=(224, 224, 6), activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=2, strides=2))
        # 卷积层2
        self.model.add(Conv2D(128, kernel_size=3, padding='same', activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=2, strides=2))
        # 卷积层3,4
        self.model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Conv2D(256, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=2, strides=2))
        # 卷积层5,6
        self.model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=2, strides=2))
        # 卷积层7,8
        self.model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Conv2D(512, kernel_size=3, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=2, strides=2))
        # 全连接层9,10,11
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))

    def get_model(self):
        self.model.summary()
        return self.model