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

dataset, ylabel = 'minst', 'Test accuracy'

# acc_cifar10 = datas[5:10]
# acc_cifar10 = datas[15:20]
acc_cifar10 = datas[25:]

# acc_cifar100 = datas[15:20]
# acc_mnist = datas[25:]


del datas
acc_cifar10_np = np.array(acc_cifar10)
acc_cifar10_np_transpose = np.transpose(acc_cifar10_np, axes=(1, 0, 2))

# acc_cifar10_6 = []
# for i in range(len(acc_cifar10[0])):   # i=0,1,..,5
#     tmp = []
#     for j in range(len(acc_cifar10)): # j=0,1,..,4
#         tmp.append([])
#
#     acc_cifar10_6.append(tmp)
del data

returnavg = []
returnstd = []
color = ['blue', 'red', 'yellow', 'olive', 'green', 'grey']
for i in range(len(acc_cifar10_np_transpose)):
    returnavg.append(np.mean(np.array(acc_cifar10_np_transpose[i]), axis=0))
    returnstd.append(np.std(np.array(acc_cifar10_np_transpose[i]), axis=0))

ax = plt.subplots(1, 1)
for i in range(6):
    plt.plot([j for j in range(1, len(returnavg[i]) + 1)], returnavg[i], color=color[i])
    r1 = list(map(lambda x: x[0] - x[1], zip(returnavg[i], returnstd[i])))
    r2 = list(map(lambda x: x[0] + x[1], zip(returnavg[i], returnstd[i])))
    plt.fill_between([j for j in range(1, len(returnavg[i]) + 1)], r1, r2, color=color[i], alpha=0.1)
    # plt.ylim(0.66, 0.74)
    # plt.ylim(0.3, 0.44)
    plt.ylim(0.986, 0.994)

plt.legend(optim_names_1)
plt.xlabel('Epoch')
plt.ylabel(ylabel)
# plt.title('Training result on {}'.format(dataset))

exp_dir = 'Plot_complex/'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
plt.savefig(os.path.join('Plot_complex', '{}_{}'.format(dataset, ylabel) + '.png'), dpi=300)

print('over')
print('love world')
