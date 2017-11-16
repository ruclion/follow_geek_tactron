from __future__ import print_function

import random
import math

PI = 3.1415926535898

f = open('test_big.xml', 'a+')


x = [3, 4.5, 2.5]
y = [6, 7.5, 1.5]


dist_start_goal_l = [1.0, 2, 3, 4, 5]
dist_start_goal_r = [2.0, 3, 4, 5, 6.0]

r_x_l = [1.3, 2]
r_x_r = [3.5, 5.5]

r_y_l = [1, 4]
r_y_r = [2.3, 8]

allcnt = 0

def randomr():
    t = random.randint(0, 1)
    return random.uniform(r_x_l[t], r_x_r[t]), random.uniform(r_y_l[t], r_y_r[t])
def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

def outxml(human_x, human_y, human_rotation_z, robot_x, robot_y, robot_rotation_z,  document):
    global allcnt
    allcnt += 1
    #<episode>
    print('<episode>', file = f)
    #<document>test1</document>
    print('<document>', document, '</document>', file = f)
    #<human_x>1</human_x>
    print('<human_x>', human_x, '</human_x>', file = f)
    #<human_y>1</human_y>
    print('<human_y>', human_y, '</human_y>', file = f)
    #<human_rotation_z>1</human_rotation_z>
    print('<human_rotation_z>', human_rotation_z, '</human_rotation_z>', file = f)
    #<robot_x>1</robot_x>
    print('<robot_x>', robot_x, '</robot_x>', file = f)
    #<robot_y>1</robot_y>
    print('<robot_y>', robot_y, '</robot_y>', file = f)
    #<robot_rotation_z>1</robot_rotation_z>
    print('<robot_rotation_z>', robot_rotation_z, '</robot_rotation_z>', file = f)
    #</episode>
    print('</episode>', file = f)
print('<?xml version="1.0" encoding="utf-8"?>', file = f)
print('<test>', file = f)

times = 0
allcnt = 0
for dist_i in range(5):
    for goal_pos_i in range(3):
        for goal_rot_i in range(4):
            human_x = x[goal_pos_i]
            human_y = y[goal_pos_i]
            human_rotation_z = goal_rot_i + 1
            cnt = 1
            while cnt < 10000:
                cnt += 1
                robot_x, robot_y = randomr()
                POSITIVEDIS = 0.8
                goal_x = human_x
                goal_y = human_y
                if human_rotation_z == 1:
                    goal_x = goal_x + POSITIVEDIS
                    #tmp_out_pos = 1
                elif human_rotation_z == 3:
                    goal_x = goal_x - POSITIVEDIS
                elif human_rotation_z == 2:
                    goal_y = goal_y + POSITIVEDIS
                else:
                    goal_y = goal_y - POSITIVEDIS
                tmp = dist(robot_x, robot_y, goal_x, goal_y)
                limit = dist(robot_x, robot_y, human_x, human_y)
                print("haha   --- ", dist_start_goal_l[dist_i], goal_pos_i, tmp)
                if tmp >= dist_start_goal_l[dist_i] and tmp <= dist_start_goal_r[dist_i] and limit > 1:
                    for robot_rot_i in range(6):
                        robot_rotation_z = 2 * PI / 6 * robot_rot_i - PI
                        # robot_rotation_z = random.uniform(-PI, PI)
                        outxml(human_x, human_y, human_rotation_z, robot_x, robot_y, robot_rotation_z, tmp)
                    break
            if cnt == 10000:
                times += 1
print('t: ', times)
print('totol :', allcnt)
print('</test>', file = f)


