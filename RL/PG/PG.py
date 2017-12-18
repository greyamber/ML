# coding: utf-8
# 这个文件实现PG方法玩平衡杆

import gym
import numpy as np
import tensorflow as tf
from Lance.lance_tensorflow import LayerManager


class PGAgent(object):
    def __init__(self, state_shape, nums, action_shape, batch_size):
        with tf.variable_scope("FP"):
            self.input_state = tf.placeholder(tf.float32, [None, state_shape], name="Agent_input")
            LM = LayerManager.LayerManager()
            L1dic = {'in_size': state_shape, 'out_size': nums,
                     'layer_type': "Linear",
                     'activation_function':tf.nn.relu,
                     'with_w': False,
                     'init': [tf.truncated_normal([state_shape, nums], mean=0.0, stddev=0.1),
                              np.zeros(nums, dtype=np.float32)]}
            L2dic = {'in_size': state_shape, 'out_size': action_shape,
                     'layer_type': "Linear",
                     'activation_function': None,
                     'with_w': False,
                     'init': [tf.truncated_normal([nums, action_shape], mean=0.0, stddev=0.1),
                              np.zeros(action_shape, dtype=np.float32)]}

            L1 = LM(self.input_state, L1dic)
            L2 = LM(L1, L2dic)
            self.logits = tf.nn.softmax(L2)
        with tf.variable_scope("BP"):
            self.labels = tf.placeholder(tf.float32, [None, action_shape])
            self.Rewards = tf.placeholder(tf.float32, [None])

            self.Loss = tf.log(tf.reduce_sum(self.logits * self.labels, 1) + 1e-8) * self.Rewards
            self.Loss = tf.reduce_sum(self.Loss) / -batch_size

            self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(self.Loss)
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

    def act(self, s):
        prob = self.sess.run(self.logits, feed_dict={self.input_state: s})
        shape = prob.shape
        action = np.random.choice(shape[1], p=prob[0])
        return action

    def train(self, s, labels, rewards):
        _, loss, logits = self.sess.run([self.train_op, self.Loss, self.logits],
                                        feed_dict={self.input_state: s, self.labels: labels,
                                        self.Rewards: rewards})


if __name__ == "__main__":
    batch_size = 8
    action_space = 2
    observation_space = 4
    train_iter = 1000
    agent = PGAgent(observation_space, 35, action_space, batch_size)
    env = gym.make('CartPole-v0')

    baseline = None
    for i in range(train_iter):
        rewards = []
        states = []
        labels = []
        for j in range(batch_size):
            s = env.reset()
            reward_num = 0.0
            reward = []
            for _ in range(env._max_episode_steps):
                if i % 100 == 0 and j == -1:
                    env.render()
                a = agent.act([s])
                label = np.zeros([action_space], dtype=np.float32)
                label[a] = 1
                labels.append(label)
                states.append(s)

                s, r, done, _info = env.step(a)
                reward_num += r
                reward.append(0.0)
                if done:
                    reward = [reward_num for _ in range(len(reward))]
                    rewards = rewards + reward
                    break
        print("Step ", i, " Mean Reward: ", np.mean(np.array(rewards)))
        rewards = np.array(rewards)
        rewards = rewards - np.mean(rewards)
        agent.train(states, labels, rewards)







