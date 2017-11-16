#from __future__ import print_function
from collections import deque

from asim_env import WebotsLidarNnEnv
import tensorflow as tf
import numpy as np
import rospy

import sys
from time import gmtime, strftime

PI = 3.14159

if __name__ == '__main__' :

    print len(sys.argv)
    if len(sys.argv) < 3:
        print "python src/env/asimmulation.py 3_1 3_2"
        print "python src/env/asimmulation.py 3_1 3_2 0" 
        exit()

    fn1 = "src/env/modes/"+sys.argv[1]
    fn2 = "src/env/modes/"+sys.argv[2]

    env = WebotsLidarNnEnv(30, 0.1)

    len_args = len(sys.argv)
    env_name = 'assim_tool.py'

    fp1 = open(fn1, "r")
    line1 = fp1.readline()
    elems = line1.split(' ')
    fp1.close()
    state = env.reset(float(elems[0]), float(elems[1]), PI*float(elems[2]))
    if len(sys.argv) > 3:
        exit()
    fp2 = open(fn2, "r")
    for lines in fp2:
        elems = lines.split(' ')
        print elems[0], elems[1]
        env.test(int(elems[0]), int(elems[1]))

    fp2.close()

    exit()

