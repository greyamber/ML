import tensorflow as tf
import numpy as np


class Initializer(object):
    def __init__(self, init_type, info=None):
        self.init_type = init_type
        self.info = info

    def set_info(self, info):
        self.info = info

    @staticmethod
    def init_PReLu_variance(node_input, node_input_next, a=0.0):
        """If a==0.0 --> ReLu"""
        if node_input is not None:
            if node_input_next is not None:
                return 4.0 / ((1 + a ** 2) * (node_input + node_input_next))
            else:
                return 2.0 / ((1 + a ** 2) * node_input)
        else:
            return 2.0 / ((1 + a ** 2) * node_input_next)

    def give_tensor(self, shape):
        if self.init_type == 'zero':
            return np.zeros(shape, dtype=np.float32)
        elif self.init_type == 'std_normal':
            return np.random.randn(*shape)
        elif self.init_type == 'truncated_normal':
            return tf.truncated_normal(shape, mean=self.info[0], stddev=self.info[1])
        elif self.init_type == 'ReLu_normal':
            return tf.truncated_normal(shape, mean=0.0,
                                       stddev=np.sqrt(self.init_PReLu_variance(self.info[0], self.info[1])))
        else:
            return None

    def __call__(self, shape, init_type=None):
        if init_type is not None:
            temp = self.init_type
            self.init_type = init_type
        tensor = self.give_tensor(shape)
        self.init_type = temp
        return tensor
