# coding: utf-8
# A3C的启动脚本

import subprocess
import os
from Lance.Lance_ML.RL.A3C.Configure import Configure

ps = subprocess.Popen(["python", "A3C.py", "-j", "ps", "-t", "0"])
workers = []
for i in range(Configure().worker_num):

    w = subprocess.Popen(["python", "A3C.py", "-j", "worker", "-t", str(i)])
    workers.append(w)

for w in workers:
    w.communicate()

for w in workers:
    w.terminate()
ps.terminate()

os.remove("Log.txt")
