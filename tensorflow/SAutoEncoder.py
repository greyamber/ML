# coding:utf-8
import tensorflow as tf
import numpy as np
import Train_iter
from Lance.lance_tensorflow import add_layer


class AutoEncoder(object):
    # num_in:输入的特征维度; num_out_list:每一层的输出特征维度; num_layer:层数,应该等于LEN（num_out_list）
    def __init__(self, num_in, num_out_list, num_layer):
        if len(num_out_list) != num_layer:
            print("Wrong num layer and num_out_list")
            exit(0)
        self.x = tf.placeholder(tf.float32, [None, num_in])
        layout = self.x
        last_out = num_in
        for i in range(num_layer):
            layout = add_layer.add_layer(layout, last_out, num_out_list[i], tf.nn.relu)
            last_out = num_out_list[i]
        self.new_x = layout
        self.layout = add_layer.add_layer(layout, last_out, num_in)

        self.Jfunc = tf.reduce_mean(tf.pow(self.layout-self.x, 2))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.Jfunc)
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def train(self, xs, times):
        TI = Train_iter.TrainIter((xs,None), 500, if_shuffle=True, has_label=False)

        for i in range(times):
            x_batch = TI.next_batch()
            self.sess.run(self.optimizer, feed_dict={self.x: x_batch})
            if (i+1) % 100 == 0:
                print(self.sess.run(self.Jfunc, feed_dict={self.x: x_batch}))

    def translate(self, xs):
        newx = self.sess.run(self.new_x, feed_dict={self.x: xs})
        return newx

    # 用于微调和最后预测, 将返还的tensor作为ＡＥ层的输出即可
    def __call__(self):
        return self.new_x

if __name__ == "__main__":
    AE = AutoEncoder(50, [30, 20], 2)
    data = np.ones([100, 50], np.float32)
    AE.train(data, 1000)
