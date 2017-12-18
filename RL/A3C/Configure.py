# coding: utf-8
# A3C的设置


import tensorflow as tf


class Configure(object):
    def __init__(self, action_len=2, obs=[4], job="worker", task="0"):
        self.action_len = action_len
        self.observation_shape = obs
        self.gama = 0.995  # r衰减
        self.hidden = 32   # 隐藏层
        self.train_iter = 1000  # 一个worker最多的更新次数
        self.lr = 1e-4

        self.update_delay = 32

        # distribute
        # worker的ip和端口（如果在不同机器上，不要使用run.py脚本）
        workers = ["127.0.0.1:1234", "127.0.0.1:1235",
                   "127.0.0.1:1236", "127.0.0.1:1237"]
        self.worker_num = len(workers)
        cluster = tf.train.ClusterSpec({"worker": workers,
                                        "ps": ["127.0.0.1:2223"]})
        self.cluster = cluster
        self.job = job
        self.task = task
