import rospy
import random

import robot_global
import motor
import time_step

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3


def get_position_res_callback(position_value):
	robot_global.position  = position_value
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

def get_rotation_res_callback(rotation_value):
	robot_global.rotation  = rotation_value
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

def robot_connect():
	rospy.Subscriber("/model_name", String)
	# connect
	model_name = None
	while model_name is None :
		try:
			model_name = rospy.wait_for_message('/model_name', String, timeout = 5)
		except:
			pass
	print ("robot %s connect success" % model_name.data)
	return model_name.data

if __name__ == '__main__' :

	rospy.init_node('webots_robot_server', anonymous = True)

	robot_global.robot_name = robot_connect()

	robot_global.PubSetPositionReq = rospy.Publisher('/simulation_set_position_req', Vector3 , queue_size = 1)
	robot_global.SubSetPositionRes = rospy.Subscriber('/simulation_set_position_res', Bool , set_position_res_callback)

	robot_global.PubResetNodePhsicsReq = rospy.Publisher('/simulation_reset_node_physics_req', Bool, queue_size = 1)
	robot_global.SubResetNodePhsicsRes = rospy.Subscriber('/simulation_reset_node_physics_res', Bool, reset_node_physics_res_callback)

	robot_global.PubGetPositionReq = rospy.Publisher('/simulation_get_position_req', Bool , queue_size = 1)
	robot_global.SubGetPositionRes = rospy.Subscriber('/simulation_get_position_res', Vector3 , get_position_res_callback)

	robot_global.PubGetRotationReq = rospy.Publisher('/simulation_get_rotation_req', Bool , queue_size = 1)
	robot_global.SubGetRotationRes = rospy.Subscriber('/simulation_get_rotation_res', Quaternion , get_rotation_res_callback)

	motor.init()
	motor.set_velocity(0,0,0,0)
	for i in range(0,3) :
		time_step.time_step_call()

	while True :
		position = Vector3()
		position.x = random.uniform(-2,2)
		position.y = 0.1
		position.z = random.uniform(-2,2)

		rotation = Quaternion()
		rotation.x = 0
		rotation.y = random.uniform(-3.14, 3.14)
		rotation.z = 0
		rotation.w = 1

		############
		done = motor.set_velocity(5,5,5,5)
		while done is False:
			time_step.time_step_call()
			done = motor.set_velocity(5,5,5,5)

		for i in range(0,10) :
			time_step.time_step_call()
		
		done = motor.set_velocity(0,0,0,0)
		while done is False:
			time_step.time_step_call()
			done = motor.set_velocity(0,0,0,0)	

		pub_reset_node_physics_req()
		while robot_global.reset_node_physics_res_flag is False:
			time_step.time_step_call()

		#############

		done = motor.set_velocity(-5,-5,-5,-5)
		while done is False:
			time_step.time_step_call()
			done = motor.set_velocity(-5,-5,-5,-5)

		for i in range(0,10) :
			time_step.time_step_call()

		done = motor.set_velocity(0,0,0,0)
		while done is False:
			time_step.time_step_call()
			done = motor.set_velocity(0,0,0,0)

		pub_reset_node_physics_req()
		while robot_global.reset_node_physics_res_flag is False:
			time_step.time_step_call()

		############

		pub_get_position_req()	
		while robot_global.get_position_res_flag is False:
			time_step.time_step_call()
		print robot_global.position
		
		pub_get_rotation_req()		
		while robot_global.get_rotation_res_flag is False:
			time_step.time_step_call()
		print robot_global.rotation
		
		'''
		for i in range(0,5) :
			time_step.time_step_call()
		'''

		'''
		pub_set_position_req(position)
		while robot_global.set_position_res_flag is False:
			time_step.time_step_call()
		'''
		