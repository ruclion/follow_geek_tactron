from __future__ import print_function
import numpy as np
import pylab as pl
import math

outf = open('444sorted_reward_label.txt', 'a+')

class loss:
    def __init__(self, txt = '', _sub = 0):
        self.txt = txt
        #self.now = _now
        self.sub = _sub
    def out(self):
        print('sub: ', self.sub, '   ', self.txt, file=outf)

def cmp1(a, b):
    if a.sub < b.sub:
        return 1
    else:
        return -1


with open('reward_out_2017-05-27-07-58-25.txt', 'r') as f:
    data = f.readlines()
print(data[0])
t = data[0].split(' ')
print(t)
print(t[23])

print(data[1])
episode = 57
cnt = []
for i, var in enumerate(data):
    t_var = var.split(' ')
    if int(t_var[2]) == episode:

        label = float(t_var[20])
        now = float(t_var[23])
        sub = float(t_var[28].split('\n')[0])
        cnt.append(loss(var, sub))
        '''
        label = float(t_var[20])
        now = float(t_var[23])
        sub = np.fabs(label - now)
        cnt.append(loss(var, sub))
        '''

print('len: ', len(cnt))
cnt.sort(cmp = cmp1)
for i, var in enumerate(cnt):
    print(i, '---', file=outf)
    var.out()
#cnt[0].out()
'''

print(data[0], data[-1])
tmpt = [1, 2, 3, 4]
print("jj ", len(tmpt[0:1]))
print("jj ", tmpt[0:1])
loss = []
iter = []
epi = []
sample_loss = []

for _line in data:
    _tmp = _line.split(' ')
    tmp = []
    for var in _tmp:
        if var != '':
            tmp.append(var)
    #print(tmp[8])
    epi.append(int(tmp[1]))
    iter.append(int(tmp[3]))
    loss.append(math.log10(float(tmp[8])))

index = 0
sum = 0
cnt = 0
x = []
for i in range(len(iter)):
    if i % 200 == 0:
        sample_loss.append(loss[i])
        x.append(i)



print(loss[8000])
print(epi[8000])
print(iter[8000])

pl.xlabel('iterator times')  # make axis labels
pl.ylabel('loss')
#pl.plot(x[101: -1], yy1[101: -1])  # use pylab to plot x and y
#pl.plot(x[101: -1], yy2[101: -1])
pl.plot(x, sample_loss)  # use pylab to plot x and y

pl.show()  # show the plot on the screen
'''