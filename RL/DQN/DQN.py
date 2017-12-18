# coding: utf-8
# 这个文件实现DQN方法玩平衡杆

import gym
import numpy as np
import tensorflow as tf
from Lance.lance_tensorflow import LayerManager


class DQNAgent(object):
    def __init__(self, observation_shape, hidden_num, action_shape, buffer_max, train_min_buffer):
        self.buffer_max = buffer_max
        self.train_min_buffer = train_min_buffer
        self.action_shape = action_shape
        self.reply_buffer = [[],[],[],[]]  # s, a, r, v_s_
        with tf.variable_scope("Input"):
            self.input_state = tf.placeholder(dtype=tf.float32,
                                              shape=[None, observation_shape], name="Input_state")
            LM = LayerManager.LayerManager()
            # no cnn:
            self.input_state = tf.reshape(self.input_state, [-1, np.prod(observation_shape)])
        with tf.variable_scope("target_Net"):
            L1dic = {'in_size': observation_shape, 'out_size': hidden_num,
                     'layer_type': "Linear",
                     'activation_function':tf.nn.relu,
                     'with_w': False,
                     'init': [tf.truncated_normal([observation_shape, hidden_num],
                                                  mean=0.0, stddev=0.1),
                              np.zeros(hidden_num, dtype=np.float32)]}
            L2dic = {'in_size': hidden_num, 'out_size': action_shape,
                     'layer_type': "Linear",
                     'activation_function': None,
                     'with_w': False,
                     'init': [tf.truncated_normal([hidden_num, action_shape], mean=0.0, stddev=0.1),
                              np.zeros(action_shape, dtype=np.float32)]}
            L1 = LM(self.input_state, L1dic)

            self.Q_target = LM(L1, L2dic)
        with tf.variable_scope("eval_Net"):
            L1dic = {'in_size': observation_shape, 'out_size': hidden_num,
                     'layer_type': "Linear",
                     'activation_function':tf.nn.relu,
                     'with_w': False,
                     'init': [tf.truncated_normal([observation_shape, hidden_num],
                                                  mean=0.0, stddev=0.1),
                              np.zeros(hidden_num, dtype=np.float32)]}
            L2dic = {'in_size': hidden_num, 'out_size': action_shape,
                     'layer_type': "Linear",
                     'activation_function': None,
                     'with_w': False,
                     'init': [tf.truncated_normal([hidden_num, action_shape], mean=0.0, stddev=0.1),
                              np.zeros(action_shape, dtype=np.float32)]}
            L1 = LM(self.input_state, L1dic)

            self.Q_eval = LM(L1, L2dic)
        with tf.variable_scope("eval_BP"):
            self.Q_estimate = tf.placeholder(tf.float32, shape=[None])
            self.labels = tf.placeholder(tf.float32, shape=[None, action_shape])
            choose_Q_eval = tf.reduce_sum(self.Q_eval * self.labels, axis=1)
            self.Loss = tf.reduce_mean(tf.square(choose_Q_eval - self.Q_estimate))
            self.train_op = tf.train.GradientDescentOptimizer(0.001).minimize(self.Loss)
        with tf.variable_scope("replacement"):
            graph = tf.get_default_graph()
            keys = tf.GraphKeys.TRAINABLE_VARIABLES
            target_variable = graph.get_collection(keys, "target_Net")
            eval_variables = graph.get_collection(keys, "eval_Net")
            self.replace_op = [tf.assign(t, e) for t, e in zip(target_variable, eval_variables)]

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def act(self, s, epsilon=0.5):
        if np.random.uniform() < epsilon:
            Q = self.sess.run(self.Q_eval, feed_dict={self.input_state: s})[0]
            action = np.argmax(Q)
        else:
            action = np.random.randint(0, self.action_shape)
        return action

    def add_to_buffer(self, s, a, r, s_, gama, done):
        i = 0
        for item in [s, a, r, s_]:
            if i == 3:
                if not done:
                    V_estimate = self.sess.run(self.Q_target, feed_dict={self.input_state: [s_]})[0]
                    V_estimate = np.max(V_estimate) * gama
                else:
                    V_estimate = 0.0
                item = V_estimate
            self.reply_buffer[i].append(item)
            i += 1
        if len(self.reply_buffer[0]) > self.buffer_max:
            for j in range(len(self.reply_buffer)):
                self.reply_buffer[j].pop(0)

    def random_pull_from_buffer(self, size):
        assert len(self.reply_buffer[0]) >= size, "Pull too many from buffer"
        sits = np.random.randint(0, len(self.reply_buffer[0]), size=size, dtype=int)
        states = np.array(self.reply_buffer[0])[sits]
        actions = np.array(self.reply_buffer[1])[sits]
        rewards = np.array(self.reply_buffer[2])[sits]
        V_estimate = np.array(self.reply_buffer[3])[sits]
        return states, actions, rewards, V_estimate

    def _train(self, states, actions, rewards, V_estimate, replace=False):
        Q_estimate = np.array(rewards) + V_estimate
        feed_dict = {self.input_state: states, self.Q_estimate: Q_estimate,
                     self.labels: actions}
        self.sess.run(self.train_op, feed_dict=feed_dict)
        if replace:
            self.sess.run(self.replace_op)

    def train(self, replace=False):
        if len(self.reply_buffer[0]) >= self.train_min_buffer:
            states, actions, rewards, V_estimate = self.random_pull_from_buffer(self.train_min_buffer)
            self._train(states, actions, rewards, V_estimate, replace=replace)


if __name__ == "__main__":
    max_buffer = 500
    batch_size = 64
    gama = 0.5
    action_space = 2
    observation_space = 4
    train_iter = 10000
    agent = DQNAgent(observation_space, 35, action_space, max_buffer, batch_size)
    env = gym.make('CartPole-v0')

    mean_r = [0.0]
    for i in range(train_iter):
        s = env.reset()
        reward_sum = 0.0
        replace = False
        for _ in range(env._max_episode_steps):
            if i % 100 == 0:
                #env.render()
                replace = True
            a = agent.act([s])
            s_, r, done, _info = env.step(a)
            action = [0 if m !=a else 1 for m in range(agent.action_shape)]
            agent.add_to_buffer(s, action, r, s_, gama, done)
            agent.train(replace=replace)
            s = s_
            reward_sum += r
            if done:
                if i % 100 == 0:
                    print("step ", i, " got: ", np.mean(mean_r))
                    mean_r = [0.0]
                break

        mean_r.append(reward_sum)
