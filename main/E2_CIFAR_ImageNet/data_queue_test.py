# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_queue_test
   Description :
   Author :       zhangjian
   date：          2019/1/9
-------------------------------------------------
   Change Activity:
                   2019/1/9: 创建文件
-------------------------------------------------
"""
base_folder = '../data/CIFAR10/'
output_foler = './output/'

import tensorflow as tf


def main():
    # 新建一个Session
    with tf.Session() as sess:
        # 我们要读三幅图片A.jpg, B.jpg, C.jpg
        filename = ['A.jpg', 'B.jpg', 'C.jpg']
        full_filename = [base_folder+f for f in filename]
        # string_input_producer会产生一个文件名队列
        filename_queue = tf.train.string_input_producer(full_filename, shuffle=True, num_epochs=3)
        # reader从文件名队列中读数据。对应的方法是reader.read
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
        tf.local_variables_initializer().run()
        # 使用start_queue_runners之后，才会开始填充队列
        threads = tf.train.start_queue_runners(sess=sess)
        i = 0
        while True:
            i += 1
            # 获取图片数据并保存
            image_data = sess.run(value)
            with open(output_foler+'test_%d.jpg' % i, 'wb') as f:
                f.write(image_data)


if __name__ == '__main__':
    main()