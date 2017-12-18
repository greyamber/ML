# coding:utf-8
# ac打平衡杆

import gym
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class Configure(object):
    def __init__(self, action_len=2, obs=[4]):
        self.action_len = action_len
        self.observation_shape = obs
        self.gama = 0.995
        self.hidden = 32
        self.train_iter = 10000
        self.lr = 1e-4

        self.update_delay = 32


class ACAgent(object):
    def __init__(self, cfg=Configure()):
        self.cfg = cfg
        self.input_state = tf.placeholder(tf.float32, [None]+cfg.observation_shape)
        # 该文件使用on policy!!! 下面注释用于备注学习
        # off-policy actor-critic
        # 近似采样分布与更新分布相同,但使用离线
        # buffer不能太大,否则梯度值相差太远（虽然正负方向不变）
        # 尽量不用lr自适应的梯度下降！因为希望每个权值的学习率尽量小
        # s, a, r, v(如果critic更新速度快于buffer更新速度，v_s_和v是误差比较大的项)
        # 不过dqn使用该策略问题不大
        self.buffer = [[],[],[],[]]
        self.pointer = 0

        with tf.variable_scope("Common_NN"):
            L1 = tf.layers.dense(self.input_state, cfg.hidden, use_bias=True, activation=tf.nn.relu)
            L2 = tf.layers.dense(L1, cfg.hidden, use_bias=True, activation=tf.nn.relu)
            self.cnn_out = L2

        with tf.variable_scope("actor"):
            # FP
            actor_out = tf.layers.dense(self.cnn_out, cfg.action_len, None, True)
            self.logits = actor_out
            self.actor_out = tf.nn.softmax(actor_out)
            # BP
            self.actor_labels = tf.placeholder(tf.float32, [None, cfg.action_len])
            self.actor_advantage = tf.placeholder(tf.float32, [None])
            choose_act = tf.reduce_sum(self.actor_out * self.actor_labels, axis=1)
            actor_loss = -tf.reduce_sum(self.actor_advantage * tf.log(choose_act + 1e-6))
            # 信息熵：信息熵越大，不确定度越大——>概率趋于均匀分布——>更新更慢,结构风险小,不容易局部收敛
            entropy = - tf.reduce_mean(tf.log(choose_act + 1e-6) * self.actor_advantage)
            # debug
            self.actor_loss = actor_loss
            self.entropy = entropy

        with tf.variable_scope("critic"):
            critic_out = tf.layers.dense(self.cnn_out, 1, None, True)
            self.critic_out = critic_out
            self.vf_next = tf.placeholder(tf.float32, shape=[None])
            self.Rewards = tf.placeholder(tf.float32, shape=[None])
            # v(s) = E[r + gama*v(s')|s] ≈ r + gama*r' + ....
            # 这个条件下：Q(s,a) = E[r + gama*v(s')|s,a] = ...
            # A(s,a) = Q - V = E[r|s,a - r|s] ≈ E[r|s,a] + gama*v(s') - v(s) = ...
            td_err = self.critic_out - self.cfg.gama*self.vf_next - self.Rewards
            critic_loss = tf.reduce_mean(tf.square(td_err))
            self.critic_loss = critic_loss

        with tf.variable_scope("update"):
            self.loss = loss = actor_loss + 0.5*critic_loss - 1e-4*entropy
            optimizer = tf.train.GradientDescentOptimizer(cfg.lr)
            # g_v = optimizer.compute_gradients(loss, tf.trainable_variables())
            # g_v = [(tf.clip_by_value(gv[0], -1., 1.), gv[1]) for gv in g_v]
            self.train_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def act(self, s):
        prob, value, logits, cnn = self.sess.run([self.actor_out, self.critic_out, self.logits, self.cnn_out],
                                                 feed_dict={self.input_state: [s]})
        action = np.random.choice(self.cfg.action_len, p=prob[0])
        return action, value[0][0]

    def train(self, s, a, r, v, done, update):
        self.buffer[0].append(s)
        self.buffer[1].append([0 if i != a else 1 for i in range(self.cfg.action_len)])
        self.buffer[2].append(r)
        self.buffer[3].append(v)

        if not (done or update):
            return

        v_next = [0 for _ in range(len(self.buffer[0]))]
        t_r = 0.0
        for i in range(len(self.buffer[0])):
            v_next[-i-1] = t_r
            t_r += self.buffer[2][-i-1]
            t_r *= self.cfg.gama

        labels = np.array(self.buffer[1], np.float32)
        states = np.array(self.buffer[0], np.float32)
        rewards = np.array(self.buffer[2], np.float32)
        value = np.array(self.buffer[3], np.float32)
        v_next = np.array(v_next, np.float32)
        self.buffer = [[],[],[],[]]
        # Q用r+gama*r'+...估计
        advantages = rewards + v_next - value
        feed_dict = {self.input_state: states, self.actor_labels: labels,
                     self.actor_advantage: advantages, self.Rewards: rewards,
                     self.vf_next: v_next}
        _, loss, closs = self.sess.run([self.train_op, self.loss, self.critic_loss], feed_dict=feed_dict)
        return closs


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    cfg = Configure()

    agent = ACAgent(cfg)
    train_iter = cfg.train_iter
    mean_r = [0.0]
    critic_loss = []
    k = 1
    for i in range(train_iter):
        s = env.reset()
        reward_sum = 0.0
        replace = False
        for _ in range(env._max_episode_steps):
            if i % 100 == 0:
                env.render()
                pass
            # s = np.array(s, dtype=np.float32)
            a, v = agent.act(s)
            s_, r, done, _info = env.step(a)
            if k % agent.cfg.update_delay == 0:
                closs = agent.train(s, a, r, v, done, True)
                k = 1
            else:
                closs = agent.train(s, a, r, v, done, False)
            if closs is not None:
                critic_loss.append(closs)
            k += 1
            s = s_
            reward_sum += r
            if done:
                break

        if i % 100 == 0:
            print_mean = np.mean(mean_r)
            print("step ", i, " got: ", print_mean)
            if print_mean > 199:
                break
            mean_r = [0.0]
        mean_r.append(reward_sum)

    plt.plot(np.array(critic_loss))
    plt.show()
