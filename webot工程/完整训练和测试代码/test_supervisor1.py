import rospy
import random

import supervisor_basic
import supervisor_global
import time_step

from std_msgs.msg import String
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion


def connect():
    rospy.Subscriber("/supervisor_name", String)
    supervisor_name = None
    while supervisor_name is None:
        try:
            supervisor_name = rospy.wait_for_message('/supervisor_name', String, timeout=5)
        except:
            pass
    print ("supervisor %s connect success" % supervisor_name.data)
    return supervisor_name.data


def pub_set_position_res():
    # print ('set postion success')
    value = Bool()
    value.data = True
    supervisor_global.PubSetPositionRes.publish(value)
    supervisor_global.set_positition_req_flag = False


def set_position_req_callback(position_value):
    # print ('receive set postion req')
    supervisor_global.position = position_value
    supervisor_global.set_positition_req_flag = True


def pub_get_position_res(position_value):
    supervisor_global.PubGetPositionRes.publish(position_value)
    supervisor_global.get_positition_req_flag = False


def get_position_req_callback(value):
    supervisor_global.get_position_req_flag = value.data


def pub_get_rotation_res(rotation_value):
    supervisor_global.PubGetRotationRes.publish(rotation_value)
    supervisor_global.get_rotation_req_flag = False


def get_rotation_req_callback(value):
    supervisor_global.get_rotation_req_flag = value.data


def reset_node_physics_callback(value):
    supervisor_global.reset_node_physics_req_flag = value.data


def pub_reset_node_physics_res():
    value = Bool()
    value.data = True
    supervisor_global.PubResetNodePhsicsRes.publish(value)
    supervisor_global.reset_node_physics_req_flag = False


if __name__ == '__main__':

    rospy.init_node('webots_supervisor_server', anonymous=True)

    supervisor_basic.supervisor_name = connect()

    supervisor_global.SubSetPositionReq = rospy.Subscriber('/simulation_set_position_req', Vector3,
                                                           set_position_req_callback)
    supervisor_global.PubSetPositionRes = rospy.Publisher('/simulation_set_position_res', Bool, queue_size=1)

    supervisor_global.SubResetNodePhsicsReq = rospy.Subscriber('/simulation_reset_node_physics_req', Bool,
                                                               reset_node_physics_callback)
    supervisor_global.PubResetNodePhsicsRes = rospy.Publisher('/simulation_reset_node_physics_res', Bool, queue_size=1)

    supervisor_global.SubGetPositionReq = rospy.Subscriber('/simulation_get_position_req', Bool,
                                                           get_position_req_callback)
    supervisor_global.PubGetPositionRes = rospy.Publisher('/simulation_get_position_res', Vector3, queue_size=1)

    supervisor_global.SubGetRotationReq = rospy.Subscriber('/simulation_get_rotation_req', Bool,
                                                           get_rotation_req_callback)
    supervisor_global.PubGetRotationRes = rospy.Publisher('/simulation_get_rotation_res', Quaternion, queue_size=1)

    robot_name = 'CAR'
    translation_name = 'translation'
    rotation_name = 'rotation'

    for i in range(0, 3):
        time_step.supervisor_time_step_call()

    node, done = supervisor_basic.get_node_from_def(robot_name)
    while done is False:
        node, done = supervisor_basic.get_node_from_def(robot_name)
        time_step.supervisor_time_step_call()

    translation_field, done = supervisor_basic.get_field(node, translation_name)
    while done is False:
        translation_field, done = supervisor_basic.get_field(node, translation_name)
        time_step.supervisor_time_step_call()

    rotation_field, done = supervisor_basic.get_field(node, rotation_name)
    while done is False:
        rotation_field, done = supervisor_basic.get_field(node, rotation_name)
        time_step.supervisor_time_step_call()

    for i in range(0, 3):
        time_step.supervisor_time_step_call()

    while True:

        position = Vector3()
        position.x = random.uniform(-2,2)
        position.y = 0.1
        position.z = random.uniform(-2,2)

        rotation = Quaternion()
        rotation.x = 0
        rotation.y = random.uniform(-3.14, 3.14)
        rotation.z = 0
        rotation.w = 1

        node, done = supervisor_basic.get_node_from_def(robot_name)
        while done is False:
            node,done = supervisor_basic.get_node_from_def(robot_name)
            time_step.supervisor_time_step_call()

        translation_field, done = supervisor_basic.get_field(node, translation_name)
        while done is False:
            translation_field, done = supervisor_basic.get_field(node, translation_name)
            time_step.supervisor_time_step_call()

        done = supervisor_basic.set_object_position(translation_field, position)
        while done is False:
            done = supervisor_basic.set_object_position(translation_field, position)
            time_step.supervisor_time_step_call()

        done = supervisor_basic.reset_node_physics(node)
        while done is False:
            done = supervisor_basic.reset_node_physics(node)
            time_step.supervisor_time_step_call()

        for i in range(0,20) :
            time_step.supervisor_time_step_call()

