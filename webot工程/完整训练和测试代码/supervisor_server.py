'''webots supervisor'''

import rospy
import random

import supervisor_basic 
import supervisor_global
import time_step

from std_msgs.msg import String
from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion

PI=3.1415926

def connect():
	rospy.Subscriber("/supervisor_name", String)
	supervisor_name = None
	while supervisor_name is None :
		try:
			supervisor_name = rospy.wait_for_message('/supervisor_name', String, timeout = 5)
		except:
			pass
	print("supervisor %s connect success" % supervisor_name.data)
	#rospy.loginfo("supervisor %s connect success", supervisor_name.data)
	return supervisor_name.data

def pub_set_position_res():
	#print ('set postion success')
	value = Bool()
	value.data = True
	supervisor_global.PubSetPositionRes.publish(value)
	supervisor_global.set_positition_req_flag = False

def set_position_req_callback(position_value):
	#print ('receive set postion req')
	supervisor_global.newposition = position_value
	print ('receive set postion req' + str(position_value))
	supervisor_global.set_positition_req_flag = True

def pub_set_rotation_res():
	#print ('set rotation success')
	value = Bool()
	value.data = True
	supervisor_global.PubSetRotationRes.publish(value)
	supervisor_global.set_rotation_req_flag = False

def set_rotation_req_callback(rotation_value):
	#print ('receive set rotation req')
	supervisor_global.newrotation = rotation_value
	print ('receive set rotation req' + str(rotation_value))
	supervisor_global.set_rotation_req_flag = True

def pub_set_human_pose_res():
	#print ('set human pose success')
	value = Bool()
	value.data = True
	supervisor_global.PubSetHumanPoseRes.publish(value)
	supervisor_global.set_human_pose_req_flag = False

#def set_human_pose_req_callback(position_value, angle):
#	#print ('receive set postion req')
#	supervisor_global.newhumanposition = position_value
#	supervisor_global.newhumanangle = angle
#	print ('receive set human pose req' + str(position_value) + str(angle))
#	supervisor_global.set_human_pose_req_flag = True

def set_human_pose_req_callback(position_value):
	#print ('receive set postion req')
	supervisor_global.newhumanposition = position_value
	#supervisor_global.newhumanangle = angle
	print ('receive set human pose req' + str(position_value))
	supervisor_global.set_human_pose_req_flag = True

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

def pub_get_human_position_res(position_value):
	supervisor_global.PubGetHumanPositionRes.publish(position_value)
	supervisor_global.get_human_positition_req_flag = False

def get_human_position_req_callback(value):
	supervisor_global.get_human_position_req_flag = value.data

def pub_get_human_rotation_res(rotation_value):
	supervisor_global.PubGetHumanRotationRes.publish(rotation_value)
	supervisor_global.get_human_rotation_req_flag = False

def get_human_rotation_req_callback(value):
	supervisor_global.get_human_rotation_req_flag = value.data

if __name__ == '__main__' :

	rospy.init_node('webots_supervisor_server', anonymous = True)

	supervisor_basic.supervisor_name = connect()

	supervisor_global.SubSetPositionReq = rospy.Subscriber('/simulation_set_position_req', Vector3 , set_position_req_callback)
	supervisor_global.PubSetPositionRes = rospy.Publisher('/simulation_set_position_res', Bool , queue_size = 1)

	supervisor_global.SubSetRotationReq = rospy.Subscriber('/simulation_set_rotation_req', Quaternion , set_rotation_req_callback)
	supervisor_global.PubSetRotationRes = rospy.Publisher('/simulation_set_rotation_res', Bool , queue_size = 1)

	supervisor_global.SubResetNodePhsicsReq = rospy.Subscriber('/simulation_reset_node_physics_req', Bool , reset_node_physics_callback)
	supervisor_global.PubResetNodePhsicsRes = rospy.Publisher('/simulation_reset_node_physics_res', Bool , queue_size = 1)

	supervisor_global.SubGetPositionReq = rospy.Subscriber('/simulation_get_position_req', Bool , get_position_req_callback)
	supervisor_global.PubGetPositionRes = rospy.Publisher('/simulation_get_position_res', Vector3 , queue_size = 1)

	supervisor_global.SubGetRotationReq = rospy.Subscriber('/simulation_get_rotation_req',  Bool, get_rotation_req_callback)
	supervisor_global.PubGetRotationRes = rospy.Publisher('/simulation_get_rotation_res', Quaternion , queue_size = 1)

    #human node topic
	supervisor_global.SubGetHumanPositionReq = rospy.Subscriber('/simulation_get_human_position_req', Bool , get_human_position_req_callback)
	supervisor_global.PubGetHumanPositionRes = rospy.Publisher('/simulation_get_human_position_res', Vector3 , queue_size = 1)

	supervisor_global.SubGetHumanRotationReq = rospy.Subscriber('/simulation_get_human_rotation_req',  Bool, get_human_rotation_req_callback)
	supervisor_global.PubGetHumanRotationRes = rospy.Publisher('/simulation_get_human_rotation_res', Quaternion , queue_size = 1)

	supervisor_global.SubSetHumanPoseReq = rospy.Subscriber('/simulation_set_human_pose_req', Vector3 , set_human_pose_req_callback)
	supervisor_global.PubSetHumanPoseRes = rospy.Publisher('/simulation_set_human_pose_res', Bool , queue_size = 1)

	robot_name = 'CAR'
	human_name = 'HUMAN'
	translation_name = 'translation'
	rotation_name = 'rotation'

	for i in range(0,5) :
		time_step.supervisor_time_step_call()

	done = False
	node = None
	while done is False:
		node,done = supervisor_basic.get_node_from_def(robot_name)
		time_step.supervisor_time_step_call()

	done = False
	translation_field = None
	while done is False:
		translation_field, done = supervisor_basic.get_field(node, translation_name)
		time_step.supervisor_time_step_call()

	done = False
	rotation_field = None
	while done is False:
		rotation_field, done = supervisor_basic.get_field(node, rotation_name)
		time_step.supervisor_time_step_call()

	done = False
	human_node = None
	while done is False:
		human_node,done = supervisor_basic.get_node_from_def(human_name)
		time_step.supervisor_time_step_call()

	#print("supervisor getnode: %s " % human_node )

	done = False
	human_translation_field = None
	while done is False:
		human_translation_field, done = supervisor_basic.get_field(human_node, translation_name)
		time_step.supervisor_time_step_call()

	#print("supervisor getfield translation_name  ok" )

	done = False
	human_rotation_field = None
	while done is False:
		human_rotation_field, done = supervisor_basic.get_field(human_node, rotation_name)
		time_step.supervisor_time_step_call()
	#print("supervisor getfield rotation_name ok" )

	while True :
		if supervisor_global.set_positition_req_flag is True:
			
			done = supervisor_basic.set_object_position(translation_field, supervisor_global.newposition)
			while done is False:
				time_step.supervisor_time_step_call()
				done = supervisor_basic.set_object_position(translation_field, supervisor_global.newposition)
					
			#rotation = Quaternion()
			#rotation.x = 0
			#rotation.y = 1
			#rotation.z = 0
			##rotation.w = -PI #random.uniform(-3.14, 3.14)
			##rotation.w = random.uniform(-3.14, 3.14)
			#rotation.w = PI/2
			##print ("set Robot Position : " + str(supervisor_global.newposition))

			#done = supervisor_basic.set_object_ratation(rotation_field, rotation)
			#while done is False:
			#	time_step.supervisor_time_step_call()
			#	done = supervisor_basic.set_object_ratation(rotation_field, rotation)
			#print("set Robot rotation: %s" % rotation)
				
			done = supervisor_basic.reset_node_physics(node)
			while done is False:
				time_step.supervisor_time_step_call()
				done = supervisor_basic.reset_node_physics(node)

			pub_set_position_res()

		if supervisor_global.set_rotation_req_flag is True:
			
			done = supervisor_basic.set_object_rotation(rotation_field, supervisor_global.newrotation)
			while done is False:
				time_step.supervisor_time_step_call()
				done = supervisor_basic.set_object_rotation(rotation_field, supervisor_global.newrotation)
					
			#print("set Robot rotation: %s" % rotation)
				
			done = supervisor_basic.reset_node_physics(node)
			while done is False:
				time_step.supervisor_time_step_call()
				done = supervisor_basic.reset_node_physics(node)

			pub_set_rotation_res()

		if supervisor_global.set_human_pose_req_flag is True:
			done = supervisor_basic.set_object_position(human_translation_field, supervisor_global.newhumanposition)
			while done is False:
				time_step.supervisor_time_step_call()
				done = supervisor_basic.set_object_position(human_translation_field, supervisor_global.newhumanposition)
					
			rotation = Quaternion()
			rotation.x = 0
			rotation.y = 1
			rotation.z = 0
			#rotation.w = supervisor_global.newhumanangle
			rotation.w = 3.14
            #random.uniform(-3.14, 3.14)
			#print ("set human Position : " + str(supervisor_global.newhumanposition))

			done = supervisor_basic.set_object_rotation(human_rotation_field, rotation)
			while done is False:
				time_step.supervisor_time_step_call()
				done = supervisor_basic.set_object_rotation(human_rotation_field, rotation)
			#print("set Robot rotation: %s" % rotation)
				
			done = supervisor_basic.reset_node_physics(human_node)
			while done is False:
				time_step.supervisor_time_step_call()
				done = supervisor_basic.reset_node_physics(human_node)

			pub_set_human_pose_res()

		if supervisor_global.reset_node_physics_req_flag is True:

			done = supervisor_basic.reset_node_physics(node)
			while done is False:
				time_step.supervisor_time_step_call()
				done = supervisor_basic.reset_node_physics(node)

			pub_reset_node_physics_res()

		if supervisor_global.get_position_req_flag is True:
			supervisor_global.position,done = supervisor_basic.get_position(translation_field)
			while done is False:
				time_step.supervisor_time_step_call()
				supervisor_global.position, done = supervisor_basic.get_position(translation_field)

			pub_get_position_res(supervisor_global.position)

		if supervisor_global.get_rotation_req_flag is True:
			supervisor_global.rotation,done = supervisor_basic.get_rotation(rotation_field)
			while done is False:
				time_step.supervisor_time_step_call()
				supervisor_global.rotation, done = supervisor_basic.get_rotation(rotation_field)

			pub_get_rotation_res(supervisor_global.rotation)
        
        #service for human node
		if supervisor_global.get_human_position_req_flag is True:
			supervisor_global.human_position,done = supervisor_basic.get_position(human_translation_field)
			while done is False:
				time_step.supervisor_time_step_call()
				supervisor_global.human_position, done = supervisor_basic.get_position(human_translation_field)

			pub_get_human_position_res(supervisor_global.human_position)

		if supervisor_global.get_human_rotation_req_flag is True:
			supervisor_global.human_rotation,done = supervisor_basic.get_rotation(human_rotation_field)
			while done is False:
				time_step.supervisor_time_step_call()
				supervisor_global.human_rotation, done = supervisor_basic.get_rotation(human_rotation_field)

			pub_get_human_rotation_res(supervisor_global.human_rotation)

		time_step.supervisor_time_step_call()
