#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
from geometry_msgs.msg import Twist, Vector3
import numpy as np

frente = Twist(Vector3(0.2,0,0), Vector3(0,0,0))
left = Twist(Vector3(0,0,0), Vector3(0,0,np.pi/6))
para = Twist(Vector3(0,0,0), Vector3(0,0,0))

if __name__ == "__main__":
    rospy.init_node("roda_exemplo")
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=3)

    try:
    	"""
    	pub.publish(Twist(Vector3(0,0,0), Vector3(0,0,np.pi/-6)))
    	rospy.sleep(1.0)
    	pub.publish(para)
    	rospy.sleep(0.5)
    	pub.publish(frente)
        rospy.sleep(2.0)
    	pub.publish(para)
    	rospy.sleep(0.5)
    	pub.publish(Twist(Vector3(0,0,0), Vector3(0,0,np.pi/6)))
    	rospy.sleep(1.0)
    	pub.publish(para)
    	rospy.sleep(0.5)
    	"""
    	print("Init")
    	rospy.sleep(2)
        while not rospy.is_shutdown():

          pub.publish(frente)
          print("Frente")
          rospy.sleep(6.2)
          pub.publish(para)
          print("Para")
          rospy.sleep(0.5)
          pub.publish(left)
          print("Vira")
          rospy.sleep(3)
          pub.publish(para)
          print("Para2")
          rospy.sleep(0.5)








    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")