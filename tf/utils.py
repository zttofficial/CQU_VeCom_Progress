# @Author: WL
# @Date  :  2021/01/14
# coding:utf-8

'''
定义一些工具函数
'''
from PIL import Image
import os
import shutil
import tensorflow as tf
import numpy as np
# from sklearn.metrics import f1_score, recall_score, precision_score

def create_exp_dir(path, scripts_to_save=None):
    '''
    创建一个文件夹，用以保存当前实验的日志及其他必要文件
    :param path: 文件夹路径
    :param scripts_to_save:需要保存的其他文件
    :return: None
    '''
    # 如果文件夹不存在，则创建对应文件夹
    if not os.path.exists(path):
        os.mkdir(path)
    print("Experiment dir : {}".format(path))
    # 如果有其他文件需要存储
    if scripts_to_save is not None:
        # 在此文件夹下创建一个新文件夹scripts
        os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            # 将源文件copy到当前目录下
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def apply_gpu_mem(gpu_num=0):
    '''
    用于动态申请GPU显存
    :param gpu_num: 用到的gpu号，一般为单卡，默认为0号GPU
    :return: None
    '''
    # 设置仅gpu_num号GPU可见
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    # 设置动态申请GPU显存
    # 获取可用gpu列表
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    assert len(gpus) > 0, "Not enough GPU hardware devices available"
    # 设置gpu仅在需要时申请显存空间
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)

# class Metrics(tf.keras.callbacks.Callback):
#     def __init__(self, valid_data):
#         super(Metrics, self).__init__()
#         self.validation_data = valid_data
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
#         val_targ = self.validation_data[1]
#         if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
#             val_targ = np.argmax(val_targ, -1)
#         _val_f1 = f1_score(val_targ, val_predict, average='macro')
#         _val_recall = recall_score(val_targ, val_predict, average='macro')
#         _val_precision = precision_score(val_targ, val_predict, average='macro')
#         logs['val_f1'] = _val_f1
#         logs['val_recall'] = _val_recall
#         logs['val_precision'] = _val_precision
#         print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
#         return
#
# def create_f1():
#     def f1_function(y_true, y_pred):
#         y_pred_binary = tf.where(y_pred>=0.5, 1., 0.)
#         tp = tf.reduce_sum(y_true * y_pred_binary)
#         predicted_positives = tf.reduce_sum(y_pred_binary)
#         possible_positives = tf.reduce_sum(y_true)
#         return tp, predicted_positives, possible_positives
#     return f1_function
#
#
# class F1_score(tf.keras.metrics.Metric):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs) # handles base args (e.g., dtype)
#         self.f1_function = create_f1()
#         self.tp_count = self.add_weight("tp_count", initializer="zeros")
#         self.all_predicted_positives = self.add_weight('all_predicted_positives', initializer='zeros')
#         self.all_possible_positives = self.add_weight('all_possible_positives', initializer='zeros')
#
#     def update_state(self, y_true, y_pred,sample_weight=None):
#         y_true = tf.cast(y_true, dtype=tf.float32)
#         tp, predicted_positives, possible_positives = self.f1_function(y_true, y_pred)
#         self.tp_count.assign_add(tp)
#         self.all_predicted_positives.assign_add(predicted_positives)
#         self.all_possible_positives.assign_add(possible_positives)
#
#     def result(self):
#         precision = self.tp_count / self.all_predicted_positives
#         recall = self.tp_count / self.all_possible_positives
#         f1 = 2*(precision*recall)/(precision+recall)
#         return f1