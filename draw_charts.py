"""
==========================================
Copyright (C) 2020 Pattern Recognition and Machine Learning Group   
All rights reserved
Description:
Created by Li Wei at 2020/11/17 20:19
Email:1280358009@qq.com
==========================================
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

sns.set_style('whitegrid')

OriginalPath = 'data'
files = []
for dirpath, dirnames, filenames in os.walk(OriginalPath):
    # print(dirpath, dirnames, filenames)
    path = [os.path.join(dirpath, names) for names in filenames]
    files.extend(path)
data = []
# test = './data/cifar10/cifar10_loss_4.pkl'
# fr = open(test, 'rb')
# inf = pickle.load(fr)
for file in files:
    print(file)
    fr = open(file, 'rb')
    inf = pickle.load(fr)
    data.append(inf)
    fr.close()
datas = []
optim_names_1 = list(data[0].keys())
optim_names_2 = list(data[10].keys())

for m in range(len(data)):
    print(m)
    optims = list(data[m].keys())
    tmp = [data[m][optims[i]] for i in range(len(optims))]
    datas.append(tmp)
t = datas[0]




for i in range(len(datas)):
    returnavg = np.mean(np.array(datas[i]), axis=0)
    returnstd = np.std(np.array(datas[i]), axis=0)


    if i < 5:
        dataset, ylabel = 'cifar10', 'loss'
    elif i < 10:
        dataset, ylabel = 'cifar10', 'val_acc'
    elif i < 15:
        dataset, ylabel = 'cifar100', 'loss'
    elif i < 20:
        dataset, ylabel = 'cifar100', 'val_acc'
    elif i < 25:
        dataset, ylabel = 'mnist', 'loss'
    else:
        dataset, ylabel = 'mnist', 'val_acc'

    color = cm.viridis(0.7)
    f, ax = plt.subplots(1, 1)
    ax.plot([i for i in range(1, len(returnavg) + 1)], returnavg, color=color)
    r1 = list(map(lambda x: x[0] - x[1], zip(returnavg, returnstd)))
    r2 = list(map(lambda x: x[0] + x[1], zip(returnavg, returnstd)))
    ax.fill_between([i for i in range(1, len(returnavg) + 1)], r1, r2, color=color, alpha=0.3)
    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel(ylabel)
    ax.set_title('Training result on {}'.format(dataset))
    exp_dir = 'Plot/'

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    f.savefig(os.path.join('Plot', '{}_{}_{}'.format(dataset, ylabel, i) + '.png'), dpi=300)

print('love world')
