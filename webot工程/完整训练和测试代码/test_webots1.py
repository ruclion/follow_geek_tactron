import rospy

import laser
import motor
import robot_global
import time_step

from std_msgs.msg import String

def robot_connect():
    rospy.Subscriber("/model_name", String)
    # connect
    model_name = None
    while model_name is None :
        try:
            model_name = rospy.wait_for_message('/model_name', String, timeout = 5)
        except:
            pass
    rospy.loginfo("robot %s connect success", model_name.data)
    return model_name.data

rospy.init_node('webots_robot_server', anonymous=True)

robot_global.robot_name = robot_connect()

laser.init()
for i in range(0,3) :
    time_step.time_step_call()

motor.init()
motor.set_velocity(0,0,0,0)
for i in range(0,3) :
    time_step.time_step_call()

step_cnt = 0

while True :

    motor.set_velocity(3, 3, 3, 3)

    time_step.time_step_call()
    time_step.time_step_call()
    time_step.time_step_call()

    laser_data, done = laser.get_laser_scan_data()
    while done is False:
        laser_data, done = laser.get_laser_scan_data()
        time_step.time_step_call()

    step_cnt += 1

    print 'step: ' + str(step_cnt)

    time_step.time_step_call()

