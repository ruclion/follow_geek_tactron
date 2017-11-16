#from __future__ import print_function
from collections import deque
from asim_env import WebotsLidarNnEnv
import tensorflow as tf
import numpy as np
import rospy
import sys
from time import gmtime, strftime




fp=open(sys.argv[1], "r")
for lines in fp:
    print lines
fp.close()
