'''webots lidar sensor'''

import rospy
import robot_global

from webots_ros.srv import sensor_enable
from webots_ros.srv import lidar_enable_point_cloud
# from webots_ros.srv import lidar_get_layer_point_cloud
# from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import LaserScan


def laser_scan_callback(laser_data) :
	robot_global.laser_data = laser_data

def init() :
	LASER_TIME_STEP = 32
	robot_name = robot_global.robot_name

	done = True

	###  lidar service
	service_name = str(robot_name) + '/lms291/enable'
	enable_laser_service = rospy.ServiceProxy(service_name, sensor_enable)
	period = LASER_TIME_STEP

	### lidar enable
	rospy.wait_for_service(service_name, timeout = 5)
	try:
		enable_laser_service(period)
		print("laser enable sucess, laser timestep: %d" % period)
	except rospy.ServiceException , e:
		print "laser enable failed"
		done = False

	### lidar point cloud service
	service_name = str(robot_name) + '/lms291/enable_point_cloud'
	enable_laser_point_cloud_service = rospy.ServiceProxy(service_name, lidar_enable_point_cloud)
	enable = True

	### enable lidar point cloud
	rospy.wait_for_service(service_name, timeout=5)
	try:
		enable_laser_point_cloud_service(enable)
		print "laser point cloud enable sucess"
	except rospy.ServiceException, e:
		print "laser point cloud enable failed"
		done = False

	### get point cloud data topic
	topic_name = str(robot_name) + '/lms291/laser_scan/layer0'
	rospy.Subscriber(topic_name, LaserScan, laser_scan_callback)

	return done

def get_laser_scan_data ():
	robot_name = robot_global.robot_name
	topic_name = str(robot_name) + '/lms291/laser_scan/layer0'

	done = True
	layer_data = None
	try:
		layer_data = rospy.wait_for_message(topic_name, LaserScan, timeout=1)
	except Exception, e:
		print "get laser scan data failed"
		done = False
	# print layer_data.ranges

	return layer_data, done

