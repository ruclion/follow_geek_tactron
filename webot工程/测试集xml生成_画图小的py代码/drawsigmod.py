from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,10,1000)
# y=np.sin(x)
y = 1./ (1.0 + np.exp(-1.0*(x * 2 - 10.0) / 2.0))
plt.plot(x,y,label='$f(d)$',color='red',linewidth=2)
plt.xlabel('d(m)')
plt.ylabel('weight')
plt.legend()
plt.show()