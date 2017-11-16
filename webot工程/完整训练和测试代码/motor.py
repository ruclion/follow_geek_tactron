'''webots rotational motor'''

import rospy
import robot_global



from webots_ros.srv import motor_set_velocity
from webots_ros.srv import motor_set_position


## init four motors : need set position to inf first
def init() :
	robot_name = robot_global.robot_name
	position_init = float('+inf')
	done = True

	####
	service_name = str(robot_name) + '/front_left_wheel/set_position'
	front_left_wheel_service = rospy.ServiceProxy(service_name, motor_set_position)

	rospy.wait_for_service(service_name, timeout = 5)
	try:
		front_left_wheel_service(position_init)
		#print "fornt_left_motor pos: %f" % self.front_left_pos.position
	except rospy.ServiceException , e:
		print "fornt_left_motor pos control service failed"
		done = False


	####
	service_name = str(robot_name) + '/back_left_wheel/set_position'
	back_left_wheel_service = rospy.ServiceProxy(service_name, motor_set_position)

	rospy.wait_for_service(service_name, timeout = 5)
	try:
		back_left_wheel_service(position_init)
	except rospy.ServiceException , e:
		print "back_left_motor pos control service failed"
		done = False

	###
	service_name = str(robot_name) + '/front_right_wheel/set_position'
	front_right_wheel_service = rospy.ServiceProxy(service_name, motor_set_position)

	rospy.wait_for_service(service_name, timeout = 5)
	try:
		front_right_wheel_service(position_init)
	except rospy.ServiceException , e:
		print "front_right_motor pos control service failed"
		done = False

	###
	service_name = str(robot_name) + '/back_right_wheel/set_position'
	back_right_wheel_service = rospy.ServiceProxy(service_name, motor_set_position)

	rospy.wait_for_service(service_name, timeout = 5)
	try:
		back_right_wheel_service(position_init)
	except rospy.ServiceException , e:
		print "back_right_motor pos control service failed"
		done = False

	return done


def set_velocity(front_left_vel, back_left_vel, front_right_vel, back_right_vel):
	robot_name = robot_global.robot_name
	done = True

	#####
	service_name = str(robot_name) + '/front_left_wheel/set_velocity'
	front_left_wheel_service = rospy.ServiceProxy(service_name, motor_set_velocity)

	rospy.wait_for_service(service_name, timeout = 3)
	try:
		front_left_wheel_service(front_left_vel)
		#print "fornt_left_motor vel: %f" % front_left_vel
	except rospy.ServiceException , e:
		print "fornt_left_motor control service failed"
		done = False
	####
	service_name = str(robot_name) + '/back_left_wheel/set_velocity'
	back_left_wheel_service = rospy.ServiceProxy(service_name, motor_set_velocity)

	rospy.wait_for_service(service_name, timeout = 3)
	try:
		back_left_wheel_service(back_left_vel)
	except rospy.ServiceException , e:
		print "back_left_motor control service failed"
		done = False

	####
	service_name = str(robot_name) + '/front_right_wheel/set_velocity'
	front_right_wheel_service = rospy.ServiceProxy(service_name, motor_set_velocity)

	rospy.wait_for_service(service_name, timeout = 3)
	try:
		front_right_wheel_service(front_right_vel)
	except rospy.ServiceException , e:
		print "fornt_right_motor control service failed"
		done = False

	####
	service_name = str(robot_name) + '/back_right_wheel/set_velocity'
	back_right_wheel_service = rospy.ServiceProxy(service_name, motor_set_velocity)

	rospy.wait_for_service(service_name, timeout = 3)
	try:
		back_right_wheel_service(back_right_vel)
	except rospy.ServiceException , e:
		print "back_right_motor control service failed"
		done = False

	return done
