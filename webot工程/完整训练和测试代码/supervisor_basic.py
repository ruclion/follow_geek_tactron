from std_msgs.msg import Bool
from std_msgs.msg import String
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist

from webots_ros.srv import sensor_enable
from webots_ros.srv import supervisor_simulation_revert
from webots_ros.srv import supervisor_get_from_def
from webots_ros.srv import node_get_field
from webots_ros.srv import field_set_vec3f
from webots_ros.srv import field_set_vec2f
from webots_ros.srv import field_set_rotation
from webots_ros.srv import field_get_vec3f
from webots_ros.srv import field_get_vec2f
from webots_ros.srv import field_get_rotation
from webots_ros.srv import node_reset_physics
from webots_ros.srv import supervisor_set_label
from webots_ros.srv import node_get_velocity

import rospy


supervisor_name = None

def set_object_position(field, position) :
	service_name = str(supervisor_name) + '/supervisor/field/set_vec3f' 
	set_position_service = rospy.ServiceProxy(service_name, field_set_vec3f)

	done = True
	rospy.wait_for_service(service_name, timeout = 3)
	try:
		set_position_service(field, 0, position)
	except rospy.ServiceException , e:
		print "set position failed"
		done = False
	return done

		
def set_object_rotation(field, rotation) :
	service_name = str(supervisor_name) + '/supervisor/field/set_rotation'
	set_rotation_service = rospy.ServiceProxy(service_name, field_set_rotation)

	done = True
	rospy.wait_for_service(service_name, timeout = 3)
	try:
		set_rotation_service(field, 0, rotation)
	except rospy.ServiceException , e:
		print "set rotation failed"
		done = False
	return done

def get_node_from_def(def_name):
	service_name = str(supervisor_name) + '/supervisor/get_from_def'
	get_node_service = rospy.ServiceProxy(service_name, supervisor_get_from_def)

	done = True
	node = None
	rospy.wait_for_service(service_name,timeout = 3)
	try:
		node = get_node_service(def_name).node
	except rospy.ServiceException , e:
		print "get node failed"
		done = False
	
	return node, done

def get_field(node, file_name): 
	service_name = str(supervisor_name) + '/supervisor/node/get_field'
	get_field_service = rospy.ServiceProxy(service_name, node_get_field)

	done = True
	field = None
	rospy.wait_for_service(service_name, timeout = 3)
	try:
		field = get_field_service(node, file_name).field
	except rospy.ServiceException , e:
		print "get field failed"
		done = False

	return field, done

def get_position(field):
	service_name = str(supervisor_name) + '/supervisor/field/get_vec3f' 
	get_position_service = rospy.ServiceProxy(service_name, field_get_vec3f)

	done = True
	rospy.wait_for_service(service_name, timeout = 3)
	try:
		position = get_position_service(field ,0).value
	except rospy.ServiceException , e:
		print "get position failed"
		done = False
		
	return position, done

def get_rotation(field):
	service_name = str(supervisor_name) + '/supervisor/field/get_rotation'
	get_rotation_service = rospy.ServiceProxy(service_name, field_get_rotation)

	done = True
	rospy.wait_for_service(service_name, timeout = 3)
	try:
		rotation = get_rotation_service(field ,0).value
	except rospy.ServiceException , e:
		print "get rotation failed"
		done = False

	return rotation, done

def reset_node_physics(node):
	service_name = str(supervisor_name) + '/supervisor/node/reset_physics'
	reset_node_physics_service = rospy.ServiceProxy(service_name, node_reset_physics)

	done = True

	rospy.wait_for_service(service_name)
	try:
		reset_node_physics_service(node)
	except rospy.ServiceException, e:
		print "set node physics failed"
		done = False

	return done

