'''webots time_step'''

from webots_ros.srv import robot_set_time_step
import rospy
import robot_global
import supervisor_basic

# def robot_initial():
#
#     robot_name = robot_global.robot_name
#
#     print ("time step service init start")
#
#     done = True
#
#     service_name = str(robot_name) + '/robot/time_step'
#     period = BASIC_TIME_STEP
#     time_step_service = rospy.ServiceProxy(service_name, robot_set_time_step)
#
#     rospy.wait_for_service(service_name, timeout=1)
#     try:
#         time_step_service(period)
#         print("time step service init success, basic time step is : " % period)
#     except rospy.ServiceException, e:
#         print "time step enable failed"
#         done = False
#
#     return done

BASIC_TIME_STEP = 32

def time_step_call():

    robot_name = robot_global.robot_name
    service_name = str(robot_name) + '/robot/time_step'
    time_step_service = rospy.ServiceProxy(service_name, robot_set_time_step)
    period = BASIC_TIME_STEP

    done = False
    while done is False:
        rospy.wait_for_service(service_name, timeout=2)
        try:
            time_step_service(period)
            done = True
            # print("time step call success")
        except rospy.ServiceException, e:
            print "time step call failed"

# def supervisor_initial():
#
#     supervisor_name = supervisor_basic.supervisor_name
#     service_name = str(supervisor_name) + '/robot/time_step'
#     time_step_service = rospy.ServiceProxy(service_name, robot_set_time_step)
#     period = BASIC_TIME_STEP
#
#     print ("supervisor time step service init start")
#     done = True
#
#     rospy.wait_for_service(service_name, timeout=1)
#     try:
#         time_step_service(period)
#         print("supervisor time step service init success, basic time step is : " % period)
#     except rospy.ServiceException, e:
#         print "supervisor time step enable failed"
#         done = False
#
#     return done

def supervisor_time_step_call():

    supervisor_name = supervisor_basic.supervisor_name
    service_name = str(supervisor_name) + '/robot/time_step'
    time_step_service = rospy.ServiceProxy(service_name, robot_set_time_step)
    period = BASIC_TIME_STEP

    done = False
    while done is False:
        rospy.wait_for_service(service_name, timeout=2)
        try:
            time_step_service(period)
            done = True
            # print("supervisor time step call success")
        except rospy.ServiceException, e:
            print "supervisor time step call failed"

