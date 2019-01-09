# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     mnist_main
   Description :
   Author :       zhangjian
   date：          2019/1/9
-------------------------------------------------
   Change Activity:
                   2019/1/9: 创建文件
-------------------------------------------------
"""

from main.E1_MINST.config import config
from main.E1_MINST import save_pic
from main.E1_MINST.softmax_regression import softmax_mnist
from main.E1_MINST.convolutional import cnn_mnist

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random as rd


input_data_folder = config.input_data_folder
output_data_folder = config.output_data_folder


def cnn_example(tr_data, val_data):
    cnn_mnist_inst = cnn_mnist(epoch=2000)
    cnn_mnist_inst.fit(tr_data, val_data, batch_size=256)


def softmax_example(tr_data, val_data):
    softmax_mnist_inst = softmax_mnist()
    softmax_mnist_inst.fit(tr_data, val_data)


def load_mnist_data():
    mnist = input_data.read_data_sets(input_data_folder, one_hot=True)
    return mnist


def mnist_data_exploration(data):
    # explore some data and save a few
    d_max = data.shape[0]
    n = rd.randint(0, d_max)

    # randomly select one example and show it
    sample = data[n].reshape(28,28)
    plt.imshow(sample, cmap='gray')
    plt.show()

    # save some images for exploration
    num = 30
    save_pic.save_pic(output_data_folder, data, num)


def main():
    # 1st step, no need to download the mnist files, already in the input folder
    mnist_data = load_mnist_data()
    train_data, validate_data, test_data = mnist_data.train, mnist_data.validation, mnist_data.test

    # explore the mnist data
    # mnist_data_exploration(train_data.images)

    # use softmax with one layer of full connection
    softmax_example(train_data, validate_data)

    # use basic cnn
    cnn_example(train_data, validate_data)


if __name__ == '__main__':
    main()

