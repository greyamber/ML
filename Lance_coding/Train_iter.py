# coding: utf-8
import numpy as np


class TrainIter(object):
    # 参数：数据和标签的元组，步长，是否打散数据
    # 转化之后所有数据都会变成float32
    # 数据必须有shape属性（tensor或者ndarray）
    # 数据如果是多维的，数据的数量应该在第一维(shape[0])
    def __init__(self, data_and_label, step_size=99, if_shuffle=False):
        self.step = step_size
        self.data = data_and_label[0]
        self.label = data_and_label[1]
        self.order = np.arange(self.data.shape[0])  # 更有效的方法是将LABEL和data一起zip起来，用np的函数直接打乱
        self.max_pointer = self.data.shape[0]
        self.now_pointer = 0

        if self.now_pointer+self.step > self.max_pointer:
            self.next_pointer = self.max_pointer
        else:
            self.next_pointer = self.now_pointer + self.step

        self.init_state = (self.now_pointer, self.next_pointer)

        if if_shuffle and self.now_pointer+self.step <= self.max_pointer:
            self.shuffle_data()

        self.init_state = (self.now_pointer, self.next_pointer)

    def shuffle_data(self):
        np.random.shuffle(self.order)

        self.now_pointer = 0
        if self.now_pointer+self.step > self.max_pointer:
            self.next_pointer = self.max_pointer
        else:
            self.next_pointer = self.now_pointer + self.step

    def next_batch(self):
        order_batch = self.order[self.now_pointer:self.next_pointer]
        batch = (np.array([self.data[x] for x in order_batch], np.float32),
                 np.array([self.label[x] for x in order_batch], np.float32))
        if self.next_pointer + self.step > self.max_pointer:
            self.now_pointer, self.next_pointer = self.init_state
        else:
            self.next_pointer += self.step
            self.now_pointer += self.step

        return batch

    def __call__(self):
        return self.next_batch()


if __name__ == '__main__':
    data = np.random.random_integers(0,10,[100000,3])
    label = np.random.random_integers(0,10,[100000,1])

    ti = TrainIter((data, label), 5000, if_shuffle=True)
    for i in range(10):
        ti()
        if i % 1 == 0:
            print(ti()[0].shape)
