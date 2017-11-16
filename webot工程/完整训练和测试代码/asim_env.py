import rospy
import random
import numpy as np

import robot_global
import human
import motor
import laser
import time_step
import math
import tf

from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3

PI=3.1415926

def get_human_position_res_callback(position_value):
    human.position = position_value
    human.get_position_res_flag = True


def pub_get_human_position_req():
    value = Bool()
    value.data = True
    human.PubGetPositionReq.publish(value)
    human.get_position_res_flag = False

def get_human_rotation_res_callback(rotation_value):
    human.rotation = rotation_value
    human.get_rotation_res_flag = True


def pub_get_human_rotation_req():
    value = Bool()
    value.data = True
    human.PubGetRotationReq.publish(value)
    human.get_rotation_res_flag = False


def get_position_res_callback(position_value):
    robot_global.position = position_value
    robot_global.get_position_res_flag = True


def pub_get_position_req():
    value = Bool()
    value.data = True
    robot_global.PubGetPositionReq.publish(value)
    robot_global.get_position_res_flag = False


def set_position_res_callback(value):
    robot_global.set_position_res_flag = value.data

def pub_set_position_req(position_value):
    robot_global.PubSetPositionReq.publish(position_value)
    robot_global.set_position_res_flag = False

def set_rotation_res_callback(value):
    robot_global.set_rotation_res_flag = value.data

def pub_set_rotation_req(rotation_value):
    robot_global.PubSetRotationReq.publish(rotation_value)
    robot_global.set_rotation_res_flag = False

def get_rotation_res_callback(rotation_value):
    robot_global.rotation = rotation_value
    robot_global.get_rotation_res_flag = True


def pub_get_rotation_req():
    value = Bool()
    value.data = True
    robot_global.PubGetRotationReq.publish(value)
    robot_global.get_rotation_res_flag = False


def reset_node_physics_res_callback(value):
    robot_global.reset_node_physics_res_flag = value.data


def pub_reset_node_physics_req():
    value = Bool()
    value.data = True
    robot_global.PubResetNodePhsicsReq.publish(value)
    robot_global.reset_node_physics_res_flag = False


def human_connect():
    rospy.Subscriber("/human_name", String)
    # connect
    model_name = None
    while model_name is None:
        try:
            model_name = rospy.wait_for_message('/human_name', String, timeout=5)
        except:
            pass
    print("human %s connect success" % model_name.data)
    return model_name.data

def robot_connect():
    rospy.Subscriber("/model_name", String)
    # connect
    model_name = None
    while model_name is None:
        try:
            model_name = rospy.wait_for_message('/model_name', String, timeout=5)
        except:
            pass
    print("robot %s connect success" % model_name.data)
    return model_name.data


MAX_LASER_RANGE = 5
MIN_LASER_RANGE = 0
COLLISION_RANGE_THRESHOLD = 0.6

class WebotsLidarNnEnv():
    def __init__(self, laser_dim, collision_threshold):

        self.laser_dim = laser_dim 
        self.collision_threshold = collision_threshold

        rospy.init_node('webots_env', anonymous=True)

        robot_global.robot_name = robot_connect()
        human.human_name = human_connect()

        robot_global.PubSetPositionReq = rospy.Publisher('/simulation_set_position_req', Vector3, queue_size=1)
        robot_global.SubSetPositionRes = rospy.Subscriber('/simulation_set_position_res', Bool,
                                                          set_position_res_callback)
                                                          
        robot_global.PubSetRotationReq = rospy.Publisher('/simulation_set_rotation_req', Quaternion, queue_size=1)
        robot_global.SubSetRotationRes = rospy.Subscriber('/simulation_set_rotation_res', Bool,
                                                          set_rotation_res_callback)

        robot_global.PubResetNodePhsicsReq = rospy.Publisher('/simulation_reset_node_physics_req', Bool, queue_size=1)
        robot_global.SubResetNodePhsicsRes = rospy.Subscriber('/simulation_reset_node_physics_res', Bool,
                                                              reset_node_physics_res_callback)

        robot_global.PubGetPositionReq = rospy.Publisher('/simulation_get_position_req', Bool, queue_size=1)
        robot_global.SubGetPositionRes = rospy.Subscriber('/simulation_get_position_res', Vector3,
                                                          get_position_res_callback)

        robot_global.PubGetRotationReq = rospy.Publisher('/simulation_get_rotation_req', Bool, queue_size=1)
        robot_global.SubGetRotationRes = rospy.Subscriber('/simulation_get_rotation_res', Quaternion,
                                                          get_rotation_res_callback)

        human.PubGetPositionReq = rospy.Publisher('/simulation_get_human_position_req', Bool, queue_size=1)
        human.SubGetPositionRes = rospy.Subscriber('/simulation_get_human_position_res', Vector3,
                                                          get_human_position_res_callback)

        human.PubGetRotationReq = rospy.Publisher('/simulation_get_human_rotation_req', Bool, queue_size=1)
        human.SubGetRotationRes = rospy.Subscriber('/simulation_get_human_rotation_res', Quaternion,
                                                          get_human_rotation_res_callback)

        for i in range(0, 5):
            time_step.time_step_call()

        motor.init()
        motor.set_velocity(0, 0, 0, 0)

        time_step.time_step_call()
        time_step.time_step_call()
        time_step.time_step_call()

        laser.init()
        #laser.get_laser_scan_data()

        self.reward_range = (-np.inf, np.inf)

        self.action_history1 = 0
        self.action_history2 = 0
        self.action_history3 = 0

        self.goal_x = 0
        self.goal_z = 0

        self.dis_x = None
        self.dis_z = None

        self.dis_x_old = None
        self.dis_z_old = None

        #goal state
        self.goalstate = 0

        for i in range(0, 5):
            time_step.time_step_call()
            
        #get robot pose
        pub_get_position_req()
        while robot_global.get_position_res_flag is False:
                time_step.time_step_call()
        pub_get_rotation_req()
        while robot_global.get_rotation_res_flag is False:
                time_step.time_step_call()

        #get human pose
        pub_get_human_position_req()
        while human.get_position_res_flag is False:
                time_step.time_step_call()
        pub_get_human_rotation_req()
        while human.get_rotation_res_flag is False:
                time_step.time_step_call()
            
    def test(self, action, times):
        if action == 0:
            motor.set_velocity(5.0, 5.0, 5.0, 5.0)
        elif action == 1:
            motor.set_velocity(7.0, 7.0, 3.0, 3.0)
        elif action == 2:
            motor.set_velocity(3.0, 3.0, 7.0, 7.0)
        elif action == 3:
            motor.set_velocity(3.0, 3.0, -3.0, -3.0)
        elif action == 4:
            motor.set_velocity(-3.0, -3.0, 3.0, 3.0)
        #laser_data, done = laser.get_laser_scan_data()
        #while done is False:
        #    laser_data, done = laser.get_laser_scan_data()
        for m in range(0, times):
            time_step.time_step_call()
        #print action

    def reset(self, px, pz, rw):
        ### room
        position = Vector3()
        position.x = px #4.1
        position.y = 0.1
        position.z = pz #2.0 #2.8

        #position = Vector3()
        #position.x = 0.0
        #position.y = 0.1
        #position.z = 0.0

        rotation = Quaternion()
        rotation.x = 0
        #rotation.y = random.uniform(-3.14, 3.14)
        rotation.y = 1
        rotation.z = 0
        rotation.w = rw #PI*3/4+random.uniform(0, PI/2) #-3.14, 3.14)  #-PI*4/5



        done = motor.set_velocity(0, 0, 0, 0)
        while done is False:
            time_step.time_step_call()
            done = motor.set_velocity(0, 0, 0, 0)

        pub_reset_node_physics_req()
        while robot_global.reset_node_physics_res_flag is False:
            time_step.time_step_call()

        pub_set_position_req(position)
        while robot_global.set_position_res_flag is False:
            time_step.time_step_call()

        pub_set_rotation_req(rotation)
        while robot_global.set_rotation_res_flag is False:
            time_step.time_step_call()

        laser_data, done = laser.get_laser_scan_data()
        while done is False:
            laser_data, done = laser.get_laser_scan_data()
            time_step.time_step_call()

        self.action_history1 = 0
        self.action_history2 = 0
        self.action_history3 = 0

        action_history = [self.action_history1, self.action_history2, self.action_history3]
        
        #get human pose
        pub_get_human_position_req()
        while human.get_position_res_flag is False:
                time_step.time_step_call()
        pub_get_human_rotation_req()
        while human.get_rotation_res_flag is False:
                time_step.time_step_call()       
                
        return 0 #np.asarray(state)
