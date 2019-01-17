# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     cifar10_extract_test
   Description :
   Author :       zhangjian
   date：          2019/1/17
-------------------------------------------------
   Change Activity:
                   2019/1/17: 创建文件
-------------------------------------------------
"""

#coding: utf-8
# 导入当前目录的cifar10_input，这个模块负责读入cifar10数据
from main.E2_CIFAR_ImageNet import cifar10_input
# 导入TensorFlow和其他一些可能用到的模块。
import tensorflow as tf
import os
import scipy.misc
from main.E2_CIFAR_ImageNet.config import config


data_path = config.input_data_folder
output_path = config.output_data_folder


def inputs_origin(data_dir):
    # filenames一共5个，从data_batch_1.bin到data_batch_5.bin
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in range(1, 6)]
    # 判断文件是否存在
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    # 将文件名的list包装成TensorFlow中queue的形式
    filename_queue = tf.train.string_input_producer(filenames)
    # cifar10_input.read_cifar10是事先写好的从queue中读取文件的函数
    # 返回的结果read_input的属性uint8image就是图像的Tensor
    read_input = cifar10_input.read_cifar10(filename_queue)
    # 将图片转换为实数形式
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    # 返回的reshaped_image是一张图片的tensor
    # 我们应当这样理解reshaped_image：每次使用sess.run(reshaped_image)，就会取出一张图片
    return reshaped_image


def main():
    with tf.Session() as sess:
        reshaped_image = inputs_origin(os.path.join(data_path, 'cifar-10-batches-bin'))

        threads = tf.train.start_queue_runners(sess=sess)
        sess.run(tf.global_variables_initializer())

        for i in range(10):
            image_array = sess.run(reshaped_image)
            scipy.misc.toimage(image_array).save(os.path.join(output_path, '%d.jpg' % i))


if __name__ == '__main__':
    main()