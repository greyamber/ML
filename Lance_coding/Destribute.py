# coding: utf-8
# 本模块实现基本的分布式通信功能，目前以utf-8字符编码作为传递数据
# 使用首部信息来控制字节数

import socket


class Server(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((cfg.server_ip, cfg.bind_port))
        self.client_count = 0
        self.pointer = None  # Server总是向该变量指向的slave发送指令
        self.socks = []  # 储存每个slaver的套接字
        self.addrs = []
        self.name_sock = []

    def find_slave(self):
        """
        实例化Server后首先调用该函数查询可用的slave,slave达到预定数才有开始下一步工作
        :return: None
        """
        self.server_socket.listen(self.cfg.clients_num)
        while self.client_count < self.cfg.clients_num:
            sock, addr = self.server_socket.accept()
            self.socks.append(sock)
            self.addrs.append(addr)
            self.client_count += 1

        for sock in self.socks:
            name = sock.recv(256).decode('utf-8')
            self.name_sock.append((name, sock))
            print("Now we have slave " + name)
            sock.send("Connected".encode())
        print("All Client Have Connected~~Slaving them!!\n")
        self.pointer = 0

    # Can always be rewrite
    # 如有需要 重写该函数
    # 一次传输的字节不能大于 256 位十进制数大小
    def send_command(self, command):
        length = str(len(command)).encode()
        while len(length) < 256:
            length = '0'.encode() + length

        self.socks[self.pointer].send(length + command.encode())
        print("Get Command: " + str(command))
        print("Send to Slave " + str(self.pointer) + '\n')

    def next_slave(self):
        """
        操作下一个slave
        如有需要，使用set_pointer()来操作任意slave
        如有需要，操作server.socks套接字数组来操作slave
        :return: 
        """
        self.pointer += 1
        if self.pointer >= len(self.socks):
            self.pointer = 0
        print("Next Slave is " + str(self.pointer) + '\n')

    def set_pointer(self, id_of_slave):
        self.pointer = id_of_slave

    def release_slave(self, pointer):
        command = "Now you are free"
        length = str(len(command)).encode()
        while len(length) < 256:
            length = '0'.encode() + length
        self.socks[pointer].send(length + command.encode())
        self.socks[pointer].close()

    def release_all(self):
        for sock in self.socks:
            command = "Now you are free"
            length = str(len(command)).encode()
            while len(length) < 256:
                length = '0'.encode() + length
            sock.send(length + command.encode())
            sock.close()

    def receive_rewards(self):
        rewards = []
        for name, sock in self.name_sock:
            length = int(sock.recv(256).decode('utf-8'))
            r = sock.recv(length).decode('utf-8')
            print(name + " gives reward: " + r)
            rewards.append(r)
        print("Received all: " + str(rewards) + '\n')
        return rewards

    def receive_reward(self):
        """
        调用该函数来一次接受一个slave的一次回报
        :return: 回报(可重写slave来控制)
        """
        sock = self.socks[self.pointer]
        length = int(sock.recv(256).decode('utf-8'))
        return sock.recv(length).decode('utf-8')


class Client(object):
    def __init__(self, cfg, name):
        self.cfg = cfg
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.name = name

    def connect(self):
        """
        首先调用该函数来建立链接
        :return: None
        """
        print("Trying to connect Server")
        self.client_socket.connect((self.cfg.server_ip, self.cfg.bind_port))
        self.client_socket.send(self.name.encode())
        print(self.client_socket.recv(256).decode('utf-8'))

    def receive_command(self):
        length = int(self.client_socket.recv(256).decode('utf-8'))
        ret = self.client_socket.recv(length).decode('utf-8')
        print("Receive command: " + str(ret))
        return ret

    def send_reward(self, reward):
        length = str(len(reward)).encode()
        while len(length) < 256:
            length = '0'.encode() + length
        print("Sending: " + str(reward))
        self.client_socket.send(length + reward.encode())

    def longing_freedom(self):
        length = int(self.client_socket.recv(256).decode('utf-8'))
        ret = self.client_socket.recv(length).decode('utf-8')
        if ret == 'Now you are free':
            return True



