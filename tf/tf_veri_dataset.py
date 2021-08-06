# @Author: WL
# @Date  :  2020/12/07
# coding:utf-8
import csv
import random
import tensorflow as tf

# Alex,Vgg,ResNet
RESIZE_TUPLE = [224,224]

def load_and_preproc_from_path_label(path1, path2, label):
    '''
    数据处理
    :param path1:
    :param path2:
    :param label:
    :return:
    '''
    # 从文件路径读取图像
    image1 = tf.io.read_file(path1)
    image2 = tf.io.read_file(path2)
    #
    image1 = tf.image.decode_jpeg(image1, channels=3)
    image2 = tf.image.decode_jpeg(image2, channels=3)
    # resize图像为224，224，3
    image1 = tf.image.resize(image1, RESIZE_TUPLE)
    image2 = tf.image.resize(image2, RESIZE_TUPLE)
    # 按照通道合并两张图像
    image_combined = tf.concat([image1, image2], 2)
    # 将label类型强制转换为int32
    label = tf.cast(label, dtype=tf.int32)
    # label = tf.compat.v1.to_int32(label)
    return image_combined, label

def make_tf_veri_dataset(csv_name, batch_size=32, num_prefetch=1):
    '''
    生成训练数据集和测试数据集
    :param csv_name: 数据集路径
    :param batch_size: 每个batch的样本数
    :param num_repeat: 样本的重复次数
    :param num_prefetch: 预加载的batch数
    :return: 训练数据集或测试数据集
    '''
    with open(csv_name, "r") as file:
        csv_reader = csv.reader(file)
        # 样本示例（相同类别为1，不同类别为0）：
        # ../../../lvhuanhuan/data/VeRi/image_train/0409_c010_00039505_0.jpg,
        # ../../../lvhuanhuan/data/VeRi/image_train/0409_c010_00076370_0.jpg,
        # 1
        all_sample_paths = [item for item in csv_reader]
    # 打乱所有样本顺序
    random.shuffle(all_sample_paths)

    # 获取image1，image2和label的索引列表
    image1_paths = [item[0] for item in all_sample_paths]
    image2_paths = [item[1] for item in all_sample_paths]
    label_paths = [int(item[2]) for item in all_sample_paths]
    print("##########################")
    print(len(image1_paths))
    print(len(label_paths))

    dataset = tf.data.Dataset.from_tensor_slices((image1_paths, image2_paths, label_paths))
    # 对数据进行预处理
    dataset = dataset.map(load_and_preproc_from_path_label,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # cache在epoch迭代过程中缓存计算结果，提升程序效率
    # 如果内存不够缓存，则使用缓存文件
    # dataset = samples_dataset.cache(filename='./cache.tf_veri_data')
    # 如果文件很大，直接用cache会导致cpu内存占用越来越高，直至内存泄漏
    # dataset = samples_dataset.cache()
    # dataset = samples_dataset
    # 先batch再repeat则两个epoch间的数据不会混在一起组成一个epoch，否则不然
    # 先shuffle再repeat则两个epoch间的数据不会混在一起shuffle，否则不然
    # 对每一个epoch的数据进行乱序，buffer_size设置为batch_size * 2大小
    dataset = dataset.shuffle(batch_size * 2)
    # 根据BATCH_SIZE构造一个batch
    dataset = dataset.batch(batch_size)
    # 预先获取下一step要load的buffer_size个batch
    dataset = dataset.prefetch(buffer_size=num_prefetch)
    # 将数据重复NUM_REPEAT次
    dataset = dataset.repeat()
    # iterator = dataset.make_one_shot_iterator()
    # features, labels = iterator.next()
    return dataset


def create_dataset(filename, batch_size=32, is_shuffle=False, is_prefetch=False,n_repeats=0):
    dataset = tf.data.TextLineDataset(filename)
    if n_repeats > 0:
        dataset.repeat(n_repeats)
    dataset = dataset.map(load_and_preproc_from_path_label)
    if is_shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    if is_prefetch:
        dataset = dataset.prefetch(buffer_size=2)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.next()
    return features, labels

if __name__ == "__main__":
    # 超参数
    BATCH_SIZE = 32
    NUM_PREFETCH = 1
    ds = make_tf_veri_dataset("veri_query.csv",
                         batch_size=BATCH_SIZE,
                         num_prefetch=NUM_PREFETCH)
    print(ds)
