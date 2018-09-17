import time

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784], name='input_ph')
    y_actual = tf.placeholder(tf.float32, shape=[None, 10], name='target_label')
    # 如果将FC层节内部初始值设的太大 会导致神经元内部的数据增长过快变成Nan
    # 如果是一个比较深的NNet 建议在其中加入 batch norm层
    # 以及在初始化的时候将初始权值的方差设定尽可能小
    W0 = tf.Variable(tf.truncated_normal([784, 400], stddev=0.1), name="w0")  # 初始化权值W
    b0 = tf.Variable(tf.constant(0.1, shape=[400]), name='b0')  # 初始化偏置项b

    W1 = tf.Variable(tf.truncated_normal([400, 200], stddev=0.1), name='w1')  # 初始化权值W
    b1 = tf.Variable(tf.constant(0.1, shape=[200]), name='b1')  # 初始化偏置项b

    W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1), name='w1')  # 初始化权值W
    b2 = tf.Variable(tf.constant(0.1, shape=[100]), name='b1')  # 初始化偏置项b

    W3 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1), name='w1')  # 初始化权值W
    b3 = tf.Variable(tf.constant(0.1, shape=[10]), name='b1')  # 初始化偏置项b

    o0 = tf.matmul(x, W0, name='layer0_mul') + b0
    i1 = tf.nn.tanh(o0, name='layer0-1_activation')

    o1 = tf.matmul(i1, W1, name='layer1_mul') + b1
    i2 = tf.nn.tanh(o1, name='layer1-2_activation')

    o2 = tf.matmul(i2, W2) + b2
    i3 = tf.nn.tanh(o2)

    o3 = tf.matmul(i3, W3) + b3
    y_predict = tf.nn.softmax(o3, name='output_softmax')  # 加权变换并进行softmax回归，得到预测概率
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_predict, labels=y_actual)  # 求交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy_mean, name='SGD_step')  # 用梯度下降法使得残差最小


    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1),
                                  name='correctness_test')  # 在测试阶段，测试准确度计算
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='trans2acc')  # 多个批次的准确度均值

    init = tf.global_variables_initializer()
    # tf.nn.conv1d()
    # set the parallel cpu cnt
    CPU_NUM = 6
    cpu_config = tf.ConfigProto(intra_op_parallelism_threads=CPU_NUM,
                                inter_op_parallelism_threads=CPU_NUM,
                                device_count={'CPU': CPU_NUM})
    with tf.Session(config=cpu_config) as sess:
        train_writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(init)
        for i in range(50000):  # 训练阶段，迭代1000次
            batch_xs, batch_ys = mnist.train.next_batch(128)  # 按批次训练，每批100行数据
            cross_entropy_val, _ = sess.run([cross_entropy_mean, train_step],
                                            feed_dict={x: batch_xs, y_actual: batch_ys})
            # print('cross', cross_entropy_val)
            # print('i1', i_)
            # print('w0', w0)
            # 执行训练
            if i % 50 == 0:
                start = time.clock()
                i_, w0, _ = sess.run([o0, W0, accuracy],
                                     feed_dict={x: mnist.test.images, y_actual: mnist.test.labels})  # 每训练100次，测试一次
                print('cost_time %f' % (time.clock() - start))
                print("accuracy:", _)
                # print('i1', i_)
                # print('w0', w0)
                # test_writer.add_summary(summary)


if __name__ == '__main__':
    main()
