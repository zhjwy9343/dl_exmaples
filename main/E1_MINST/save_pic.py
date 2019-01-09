#coding: utf-8
import scipy.misc
import os


def save_pic(save_dir, mnist, num=20):

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    # 保存前num张图片
    for i in range(num):

        image_array = mnist.train.images[i, :]
        # TensorFlow中的MNIST图片是一个784维的向量，我们重新把它还原为28x28维的图像。
        image_array = image_array.reshape(28, 28)
        # 保存文件的格式为 mnist_train_0.jpg, mnist_train_1.jpg, ... ,mnist_train_19.jpg
        filename = save_dir + 'mnist_train_%d.jpg' % i
        # 将image_array保存为图片
        # 先用scipy.misc.toimage转换为图像，再调用save直接保存。
        scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)

    print('Please check: %s ' % save_dir)

