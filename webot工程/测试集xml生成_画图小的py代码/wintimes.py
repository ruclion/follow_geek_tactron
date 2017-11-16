from __future__ import print_function
import numpy as np
import pylab as pl

with open('wintimes_18_18.txt', 'r') as f:
    data = f.readlines()

print(data[0], data[-1])

y1 = []
y2 = []
x = []
index = 0
for _line in data:
    _tmp = _line.split(' ')
    tmp = []
    for var in _tmp:
        if var != '':
            tmp.append(var)
    #print(tmp[8])
    y1.append(int(tmp[7]) / 100.0)
    tt = tmp[8].split('\n')
    y2.append(int(tt[0]) / 100.0)
    x.append(int(tmp[1]))
yy1 = []
yy2 = []
for i in range(len(x)):
    if i > 2 and i < len(x) - 2:
        yy1.append((y1[i - 2] * 1.0 + y1[i - 1] * 3.0 + y1[i] * 7 + y1[i + 1] * 3.0 + y1[i + 2] * 1.0) / 15.0)
        yy2.append((y2[i - 2] * 1.0 + y2[i - 1] * 3.0 + y2[i] * 7 + y2[i + 1] * 3.0 + y2[i + 2] * 1.0) / 15.0)
    else:
        print("aaaa")
        yy1.append(y1[i])
        yy2.append(y2[i])




pl.xlabel('episode times')  # make axis labels
pl.ylabel('rate')
#pl.plot(x[101: -1], yy1[101: -1])  # use pylab to plot x and y
#pl.plot(x[101: -1], yy2[101: -1])
pl.plot(x[101: -1], y1[101: -1])  # use pylab to plot x and y
pl.plot(x[101: -1], y2[101: -1])
pl.show()  # show the plot on the screen
