# coding:utf-8

import tensorflow as tf

class softmax_mnist():

    def __init__(self, epoch=1000):
        self.epoch = epoch


    def __softmax_model(self, x, y_):

        # W是Softmax模型的参数，将一个784维的输入转换为一个10维的输出
        # 在TensorFlow中，变量的参数用tf.Variable表示
        W = tf.Variable(tf.zeros([784, 10]))
        # b是又一个Softmax模型的参数，我们一般叫做“偏置项”（bias）。
        b = tf.Variable(tf.zeros([10]))

        # y=softmax(Wx + b)，y表示模型的输出
        y = tf.nn.softmax(tf.matmul(x, W) + b)

        # 根据y, y_构造交叉熵损失
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

        # 有了损失，我们就可以用随机梯度下降针对模型的参数（W和b）进行优化
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

        return train_step, y


    def fit(self, mnist_train, mnist_validation, batch_size=128):
        # 创建x，x是一个占位符（placeholder），代表待识别的图片
        x = tf.placeholder(tf.float32, [None, 784])

        # y_是实际的图像标签，同样以占位符表示。
        y_ = tf.placeholder(tf.float32, [None, 10])

        model, y = self.__softmax_model(x, y_)

        # 创建一个Session。只有在Session中才能运行优化步骤train_step。
        sess = tf.InteractiveSession()
        # 运行之前必须要初始化所有变量，分配内存。
        tf.global_variables_initializer().run()
        print('start training...')

        # 进行1000步梯度下降
        for _ in range(self.epoch):

            # batch_xs是形状为(100, 784)的图像数据，batch_ys是形如(100, 10)的实际标签
            # batch_xs, batch_ys对应着两个占位符x和y_
            batch_xs, batch_ys = mnist_train.next_batch(batch_size)

            # 在Session中运行train_step，运行时要传入占位符的值
            sess.run(model, feed_dict={x: batch_xs, y_: batch_ys})

        # 正确的预测结果
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # 计算预测准确率，它们都是Tensor
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 在Session中运行Tensor可以得到Tensor的值
        # 这里是获取最终模型的正确率
        print(sess.run(accuracy, feed_dict={x: mnist_validation.images, y_: mnist_validation.labels}))  # 0.9185
