from __future__ import print_function
import numpy as np
import pylab as pl

with open('train_totol_loss.txt', 'r') as f:
    data = f.readlines()

# print(data[0], data[-1])

y = []
x = []
index = 0
cnt = 0
for _line in data:
    line = _line.split('\n')
    for c in line:
        if len(c) <= 0 or c[0] == '\n' or c[0] == 'g':
            continue
        print('??', c)
        # print('ll:', _line[0])
        if _line[0] != 'g' and _line[0] != ' ':
            line = _line.split('\n')[0]
            print('fff', _line, 'fff')
            line = float(line)
            y.append(line)
            cnt += 1
            x.append(cnt)
    '''
    y.append(float(_line))
    cnt += 1
    x.append(cnt)
    '''





pl.xlabel('episode times')  # make axis labels
pl.ylabel('loss')
#pl.plot(x[101: -1], yy1[101: -1])  # use pylab to plot x and y
#pl.plot(x[101: -1], yy2[101: -1])
pl.plot(x, y)  # use pylab to plot x and y
pl.show()  # show the plot on the screen
