#coding=utf-8

import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import utils
import logging
import sys
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# tf.compat.v1.experimental.output_all_intermediates(True)
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D
import numpy as np
from tf_veri_dataset import make_tf_veri_dataset, create_dataset
from tf_dmv_dataset import make_tf_dmv_dataset
import time
from cnn import AlexNet, VGGNet_11
#from tensorflow.keras.applications import ResNet50V2, ResNet101V2, VGG16, VGG19, DenseNet121, MobileNetV2
from utils import apply_gpu_mem
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa

# 
# BATCH_SIZE = 32
# NUM_PREFETCH = 2
# NUM_EPOCHS = 50

parser = argparse.ArgumentParser("veri")
parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
parser.add_argument('--prefetch', type=int, default=1, help="num of prefetch batch(es)")
parser.add_argument('--epochs', type=int, default=30, help="num of training epochs")
parser.add_argument('--sigleepoch', type=int, default=1, help="num of training epochs")
parser.add_argument('--lr', type=float, default=0.0001, help="the learning rate")
parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
parser.add_argument('--net', type=str, default="alex", help="name of cnn")
parser.add_argument('--workers', type=int, default=4, help="num of workers of model fit")
parser.add_argument('--save_each_epoch', type=int, default=10, help="num of workers of model fit")
args = parser.parse_args()

# 创建文件夹记录实验结果
# exp_dir = 'train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
# utils.create_exp_dir(exp_dir)
#
# log_format = '%(asctime)s %(message)s'
# logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join(exp_dir, 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)

def train():
    # 设置gpu
    apply_gpu_mem(args.gpu)
    logging.info('gpu device = %d' % args.gpu)
    # 计算训练集大小
    # with open("./data_csv/veri/veri_train.csv") as file:
    #     veri_train_len = len(file.readlines())
    # with open("./data_csv/veri/veri_test.csv") as file:
    #     veri_test_len = len(file.readlines())
    veri_train_len = utils.iter_count("/mnt/train_data/compare/txt_cut/vecom_train.txt")
    veri_test_len = utils.iter_count("/mnt/train_data/compare/txt_cut/vecom_test.txt")
    # 加载数据集
    veri_train_data = make_tf_veri_dataset("/mnt/train_data/compare/txt_cut/vecom_train.txt",
                                         batch_size=args.batch_size,
                                         num_prefetch=args.prefetch,)
    # veri_valid_data = make_tf_veri_dataset("./train/test.txt",
    #                                     batch_size=args.batch_size,
    #                                     num_prefetch=args.prefetch,)
    veri_test_data = make_tf_veri_dataset("/mnt/train_data/compare/txt_cut/vecom_test.txt",
                                           batch_size=args.batch_size,
                                           num_prefetch=args.prefetch, )

    # veri_train_data = create_dataset("./data_csv/veri/veri_train.csv",
    #                                  batch_size=args.batch_size,
    #                                  is_shuffle=True,
    #                                  is_prefetch=True)
    # veri_valid_data = create_dataset("./data_csv/veri/veri_query.csv",
    #                                  batch_size=args.batch_size,
    #                                  is_shuffle=True,
    #                                  is_prefetch=True )
    # veri_test_data = create_dataset("./data_csv/veri/veri_test.csv",
    #                                 batch_size=args.batch_size,
    #                                 is_shuffle=True,
    #                                 is_prefetch=True)
    # 选择不同的网络


    if args.net == "vgg11":
        # Vgg11
        model = VGGNet_11().get_model()
    elif args.net == 'vgg16':
        model = VGG16(weights=None, input_shape=(224,224,6),classes=2)
    elif args.net == 'vgg19':
        model = VGG19(weights=None, input_shape=(224,224,6),classes=2)
    elif args.net == "res50":
        model = ResNet50V2(weights=None, input_shape=(224,224,6),classes=2)
    elif args.net == "res101":
        model = ResNet101V2(weights=None, input_shape=(224,224,6),classes=2)
    elif args.net == "dense121":
        model = DenseNet121(weights=None, input_shape=(224,224,6),classes=2)
    elif args.net == "mobile":
        model = MobileNetV2(weights=None, input_shape=(224, 224, 6), classes=2)
    else:
        # 默认为AlexNet
        model = AlexNet().get_model()

    model.compile(loss="binary_crossentropy",
                  optimizer=optimizers.Adam(args.lr),
                  metrics=['accuracy', tfa.metrics.F1Score(num_classes=3, threshold=0.5, average='micro'), tf.keras.metrics.AUC()])
    #model_dir = "./models/" + args.net
    model_dir = './models'
    f = open("./train/log.txt","w",encoding="utf-8")
    e = 0

    #model_checkpoint = ModelCheckpoint('1.hdf5', monitor='loss',verbose=0, save_best_only=True, save_freq=1)
    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=1)

    while True:
       # 训练模型
        if e>= args.epochs:
            break
        #model.load_weights(model_dir + "/" + "latest.h5")
        model.fit(veri_train_data,
                 epochs=args.sigleepoch,
                 steps_per_epoch=veri_train_len // args.batch_size,
                 # validation_data=valid_data,
                 # validation_steps=valid_len // BATCH_SIZE,
                 # 当数据集为dataset形式时不支持validation_split
                 # validation_split=0.2,
                 workers=args.workers,
                 use_multiprocessing=True,
                 verbose=1,
                 callbacks=[EarlyStopping],)
                #  callbacks=[model_checkpoint,TensorBoard(log_dir=r"./logs", write_graph=True, write_images=True,
                #             histogram_freq=1,update_freq="batch",)])
      
       # print("Model Testing")

       # model.evaluate(veri_test_data, steps=veri_test_len // args.batch_size)
       # 获取当前时间:202012082220
       #local_time = time.strftime(("%Y%m%d%H%M"), time.localtime())
       # 设置模型参数的保存路径及名称
        e = e+args.sigleepoch
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
       
        model.save_weights(model_dir + "/" + "latest.h5")
        if e%args.save_each_epoch == 0:
            model.save_weights(os.path.join(model_dir,str(e)+"_epochs.h5"))
        
        print("Model Testing")
        result = model.evaluate(veri_test_data, steps=veri_test_len // args.batch_size, verbose=1)
        # print(model.metrics_names)
        f.write(os.path.join(model_dir,str(e)+"_epochs.h5")+"   loss:"+str(result[0])+"  acc:"+str(result[1]) 
                +"  f1:"+str(result[2])+"  auc:"+str(result[3])+"\n")


if __name__ == "__main__":
    train()
