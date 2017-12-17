from __future__ import print_function
import numpy as np
import pylab as pl

with open('train_style.txt', 'r') as f:
    data = f.readlines()

# print(data)

y = []
x = []
index = 0
cnt = 0
for _line in data:
    '''
    _tmp = _line.split(' ')
    tmp = []
    for var in _tmp:
        if var != '':
            tmp.append(var)
    #print(tmp[8])
    '''
    if _line[0:3] == '[[[':
        print(float(_line[3:10]))
        y.append(float(_line[3:10]))
        cnt += 1
        x.append(cnt)





pl.xlabel('episode times')  # make axis labels
pl.ylabel('style_vec[0][0]')
#pl.plot(x[101: -1], yy1[101: -1])  # use pylab to plot x and y
#pl.plot(x[101: -1], yy2[101: -1])
pl.plot(x, y)  # use pylab to plot x and y
pl.show()  # show the plot on the screen
