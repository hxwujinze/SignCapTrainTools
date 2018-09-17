import os
import pickle
import random

import numpy as np
import tensorflow as tf

DTYPE = tf.float32
DATA_DIR_PATH = os.path.join(os.getcwd(), 'data')


class ClassifyModel:
    def __init__(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.input_tensor = tf.placeholder(shape=[None, 14, 64], name='input', dtype=DTYPE)

            with tf.variable_scope('conv1'):
                filter_w1 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            shape=[3, 14, 32],
                                            name='filter_w1_1',
                                            dtype=DTYPE, )

                tmp_out1_1 = tf.nn.conv1d(self.input_tensor,
                                          filter_w1,
                                          data_format='NCW',
                                          stride=1,
                                          padding='SAME')

                conv1_1out = tf.nn.relu(tmp_out1_1)

                filter_w1_2 = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                                              shape=[3, 32, 32],
                                              dtype=DTYPE,
                                              name='filter_w1_2', )
                tmp_out1_2 = tf.nn.conv1d(conv1_1out,
                                          filter_w1_2,
                                          data_format='NCW',
                                          stride=1,
                                          padding='SAME')
                conv_out1_2 = tf.nn.relu(tmp_out1_2)

                pool_output = tf.layers.max_pooling1d(inputs=conv_out1_2,
                                                      pool_size=3,
                                                      strides=2, padding='valid',
                                                      data_format='channels_first',
                                                      name='con1_pool')
            # end conv1
            self.output_tensor = pool_output
        tf.summary.FileWriter('./log', self.graph)

    def inference(self, input_tensor):
        self.input_tensor.assing(input_tensor)
        with tf.Session(self.graph) as sess:
            res = sess.run(fetches=[self.output_tensor], feed_dict=self.input_tensor)
            return res

    def train(self):
        data_set = load_data()

        x = tf.placeholder(DTYPE, [None, 14, 64], name='input')
        y = tf.placeholder(tf.int16, [None, 1], name='labels')


def load_data():
    data_dir = os.path.join(DATA_DIR_PATH, 'new_train_data')
    with open(data_dir, 'rb') as f:
        data_set = pickle.load(f)

    tf.data.Dataset.from_tensor_slices()
    inputs = []
    label = []
    random.shuffle(data_set)
    for each in data_set:
        data_mat = each[0].T
        data_mat = np.where(data_mat > 0.00000000001, data_mat, 0)
        inputs.append(data_mat)
        label.append(each[1])

    return inputs, label


if __name__ == '__main__':
    m = ClassifyModel()
