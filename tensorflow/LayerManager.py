# coding: utf-8
import tensorflow as tf
from Lance.lance_tensorflow import Lance_Initializer


class LayerManager(object):
    def __init__(self):
        self.help_info = '''
        layer_type:Linear
        info_dict:{in_size:int, out_size:int, activation_function(can be None):tf.nn, with_w:, init(can be None):[w,b]}

        layer_type:BN
        info_dict:None

        layer_type:Conv
        info_dict:{shape:[1,5,5,1], stride:[1,1,1,1], padding:['SAME',',MAX'], init(can be None):[w,b], forward_BN:Bool, active_func(can be None):}

        layer_type:Pooling
        info_dict:{ksize:[1,2,2,1], strides:[1,2,2,1], padding:['SAME',',MAX']}

        '''

    @staticmethod
    def BN_layer(inputs):
        batch_mean, batch_var = tf.nn.moments(inputs, list(range(len(inputs.get_shape()) - 1)), keep_dims=True)
        shift = tf.Variable(tf.zeros(batch_mean.shape))
        scale = tf.Variable(tf.ones(batch_mean.shape))
        epsilon = 1e-3
        return_args = tf.nn.batch_normalization(inputs, batch_mean, batch_var, shift, scale, epsilon)
        return return_args

    def show_help(self):
        print(self.help_info)

    def give_layer(self, inputs, layer_type='Linear', args=None):
        """
        :param inputs: a tensor
        :param layer_type: String of Layer's type,
        :param args: other param of layer
        :return return_args: Layer(inputs) and other information defined by type
        """
        return_args = None
        if layer_type == 'Linear':
            # needed args:[in_size, out_size, activation_function, with_w, init]
            in_size, out_size, activation_function, with_w, init = args
            if init is None:
                W = tf.Variable(tf.random_normal([in_size, out_size]))
                biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            else:
                W = tf.Variable(init[0])
                biases = tf.Variable(init[1])
            xW_plus_b = tf.add(tf.matmul(inputs, W), biases)
            if activation_function is None:
                return_args = xW_plus_b, W, biases
            else:
                return_args = activation_function(xW_plus_b), W, biases
            if not with_w:
                return_args = return_args[0]

        elif layer_type == 'BN':
            batch_mean, batch_var = tf.nn.moments(inputs, list(range(len(inputs.get_shape()) - 1)), keep_dims=True)
            shift = tf.Variable(tf.zeros(batch_mean.shape))
            scale = tf.Variable(tf.ones(batch_mean.shape))
            epsilon = 1e-3
            return_args = tf.nn.batch_normalization(inputs, batch_mean, batch_var, shift, scale, epsilon)

        elif layer_type == 'Conv':
            shape, stride, pading, init, forward_BN, active_func = args
            if init is None:
                W_forconv = tf.truncated_normal(shape=shape, mean=0.0, stddev=0.01)
                bias = tf.constant(1e-3, [shape[3]])
            else:
                W_forconv = init[0]
                bias = init[1]
            tempw = tf.Variable(W_forconv)
            tempb = tf.Variable(bias)
            tempconv = tf.nn.conv2d(inputs, tempw, strides=stride, padding=pading)
            if forward_BN:
                BN_out = self.BN_layer(tempconv)
                return_args = active_func(BN_out + tempb)
            else:
                return_args = active_func(tempconv + tempb)

        elif layer_type == 'Pooling':
            ksize, strides, padding = args
            return_args = tf.nn.max_pool(inputs, ksize=ksize, strides=strides, padding=padding)

        return return_args

    def __call__(self, inputs, info_dict):
        if info_dict['layer_type'] == 'Linear':
            return self.give_layer(inputs, 'Linear',
                                   args=[info_dict['in_size'], info_dict['out_size'],
                                         info_dict['activation_function'],
                                         info_dict['with_w'], info_dict['init']])

        elif info_dict['layer_type'] == 'BN':
            return self.give_layer(inputs, 'BN',
                                   args=None)

        elif info_dict['layer_type'] == 'Conv':
            return self.give_layer(inputs, 'Conv',
                                   args=[info_dict['shape'], info_dict['stride'],
                                         info_dict['padding'], info_dict['init'],
                                         info_dict['forward_BN'], info_dict['active_func']])

        elif info_dict['layer_type'] == 'Pooling':
            return self.give_layer(inputs, 'Pooling',
                                   args=[info_dict['ksize'], info_dict['strides'],
                                         info_dict['padding']])