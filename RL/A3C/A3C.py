# coding: utf-8

import gym
import argparse
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from Lance.Lance_ML.RL.A3C.Configure import Configure


class ACWorker(object):
    """
    详情见actor-critic
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.input_state = tf.placeholder(tf.float32, [None]+cfg.observation_shape)
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


class A3CAgent(object):
    """
    A3C 算法
    主要涉及tensorflow分布式搭建的几个api
    1.server
    2.device
    3.Graph and GraphKeys
    4.Session
    主要使用的几个灵活api:
    1.tf.gradient
    2.Variable.assign
    3.Graph.get_collection
    """
    def __init__(self, sess, cfg):
        self.cfg = cfg
        with tf.device('/job:ps/task:0/cpu:0'):
            with tf.variable_scope("global"):
                self.global_agent = ACWorker(cfg)

        with tf.device('/job:worker/task:'+str(cfg.task) + "/cpu:0"):
            with tf.variable_scope("local"):
                self.local_agent = ACWorker(cfg)

        self.global_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "global")
        self.local_trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "local")

        with tf.device('/job:worker/task:'+str(cfg.task) + "/cpu:0"):
            with tf.variable_scope("assign_and_update"):
                self.assign = tf.group(*[v1.assign(v2)
                                         for v1, v2 in zip(self.local_trainables,
                                                           self.global_trainables)])
                grads = tf.gradients(self.local_agent.loss, self.local_trainables)
                grads_and_vars = list(zip(grads, self.global_trainables))
                opt = tf.train.AdamOptimizer(cfg.lr)
                self.train_op = opt.apply_gradients(grads_and_vars)
            with tf.variable_scope("init"):
                all_global = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
                all_global += tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, "global")
                self.global_init = tf.variables_initializer(all_global, "global_init")
                all_local = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "local")
                all_local += tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, "local")
                all_local += tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, "assign_and_update")
                all_local += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "assign_and_update")
                self.local_init = tf.variables_initializer(all_local, "local_init")

        self.sess = sess
        if self.cfg.task == 0:
            self.sess.run(self.global_init)
            with open("Log.txt", 'a+') as file:
                file.write("Initial finished")
        self.sess.run(self.local_init)

    def act(self, s):
        prob, value = self.sess.run([self.local_agent.actor_out, self.local_agent.critic_out],
                                    feed_dict={self.local_agent.input_state: [s]})
        action = np.random.choice(self.cfg.action_len, p=prob[0])
        return action, value[0][0]

    def pull_paras(self):
        self.sess.run(self.assign)

    def train(self, s, a, r, v, done, update):
        self.local_agent.buffer[0].append(s)
        self.local_agent.buffer[1].append([0 if i != a else 1 for i in range(self.cfg.action_len)])
        self.local_agent.buffer[2].append(r)
        self.local_agent.buffer[3].append(v)

        if not (done or update):
            return

        v_next = [0 for _ in range(len(self.local_agent.buffer[0]))]
        t_r = 0.0
        for i in range(len(self.local_agent.buffer[0])):
            v_next[-i-1] = t_r
            t_r += self.local_agent.buffer[2][-i-1]
            t_r *= self.local_agent.cfg.gama

        labels = np.array(self.local_agent.buffer[1], np.float32)
        states = np.array(self.local_agent.buffer[0], np.float32)
        rewards = np.array(self.local_agent.buffer[2], np.float32)
        value = np.array(self.local_agent.buffer[3], np.float32)
        v_next = np.array(v_next, np.float32)
        self.local_agent.buffer = [[],[],[],[]]
        # Q用r+gama*r'+...估计
        advantages = rewards + v_next - value
        feed_dict = {self.local_agent.input_state: states, self.local_agent.actor_labels: labels,
                     self.local_agent.actor_advantage: advantages,
                     self.local_agent.Rewards: rewards,
                     self.local_agent.vf_next: v_next}
        _, loss, closs = self.sess.run([self.train_op,
                                        self.local_agent.loss,
                                        self.local_agent.critic_loss], feed_dict=feed_dict)
        self.pull_paras()
        return closs


def main(job, task):
    cfg = Configure(job=job, task=task)
    server = tf.train.Server(cfg.cluster, job_name=cfg.job,
                             task_index=cfg.task)

    if cfg.job == "ps":
        server.join()
    else:
        sess = tf.Session(server.target)
        agent = A3CAgent(sess, cfg)
        while True:
            try:
                with open("Log.txt", 'r') as file:
                    print(file.readline())
                    break
            except Exception as e:
                print("Waiting for initial")

        env = gym.make('CartPole-v0')
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
                    #env.render()
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
                if print_mean > 150:
                    break
                mean_r = [0.0]
            mean_r.append(reward_sum)

        if cfg.task == 0:
            plt.plot(np.array(critic_loss))
            plt.show()
        else:
            exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run commands")
    parser.add_argument('-j', '--job', default="ps", type=str, help="job")

    parser.add_argument('-t', '--task', type=int, default=0, help="task")

    args = parser.parse_args()
    main(args.job, args.task)
