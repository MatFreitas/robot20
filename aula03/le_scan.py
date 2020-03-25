#! /usr/bin/env python
# -*- coding:utf-8 -*-


import rospy
import numpy as np
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan


def scaneou(dado):
	#print("Faixa valida: ", dado.range_min , " - ", dado.range_max ) #faixa de 0.12 até 3.5#
	#print("Leituras:")
	print(np.array(dado.ranges).round(decimals=2))
	x = 0
	#print("Intensities")
	#print(np.array(dado.intensities).round(decimals=2))

	""" #TENTATIVA DE CRIACAO DE UM MECANISMO PARA EVITAR PAREDES#
	for n in range(360):
		if np.array(dado.ranges[n]).round(decimals=2) < 0.25:
			rospy.sleep(2)
			velocidade = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
			velocidade_saida.publish(velocidade)

			if n <= 90 and x == 0:
				velocidade_saida.publish(Twist(Vector3(0,0,0), Vector3(0,0,-np.pi/6)))
				rospy.sleep(1.5)
				velocidade = Twist(Vector3(-0.1, 0, 0), Vector3(0, 0, 0))
				velocidade_saida.publish(velocidade)
				print("FUGA PARA TRAS!")
				x = 1
				rospy.sleep(2)
			elif n >= 270 and x == 0:
				velocidade_saida.publish(Twist(Vector3(0,0,0), Vector3(0,0,np.pi/6)))
				rospy.sleep(1.5)
				velocidade = Twist(Vector3(-0.1, 0, 0), Vector3(0, 0, 0))
				velocidade_saida.publish(velocidade)
				print("FUGA PARA TRAS!")
				x = 1
				rospy.sleep(2)
			elif n > 90 and n <= 180 and x == 0:
				velocidade_saida.publish(Twist(Vector3(0,0,0), Vector3(0,0,np.pi/6)))
				rospy.sleep(1.5)
				velocidade = Twist(Vector3(0.1, 0, 0), Vector3(0, 0, 0))
				velocidade_saida.publish(velocidade)
				print("FUGA PARA FRENTE!")
				x = 1
				rospy.sleep(2)
			elif n > 180 and n < 270 and x == 0:
				velocidade_saida.publish(Twist(Vector3(0,0,0), Vector3(0,0,np.pi/6)))
				rospy.sleep(1.5)
				velocidade = Twist(Vector3(0.1, 0, 0), Vector3(0, 0, 0))
				velocidade_saida.publish(velocidade)
				print("FUGA PARA FRENTE!")
				x = 1
				rospy.sleep(2)
			else:
				print("ERRO!")
				x = 1
	"""
	if np.array(dado.ranges[0]).round(decimals=2) < 1:
		velocidade = Twist(Vector3(0.05, 0, 0), Vector3(0, 0, 0.2))
		velocidade_saida.publish(velocidade)
		print("Ahead Captain!")
	elif np.array(dado.ranges[0]).round(decimals=2) > 1.02:
		velocidade = Twist(Vector3(-0.05, 0, 0), Vector3(0, 0, 0.2))
		velocidade_saida.publish(velocidade)
		print("Roll back now!")
	x = 0

if __name__=="__main__":



	rospy.init_node("le_scan")
	velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 3 )
	recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)

	while not rospy.is_shutdown():
		rospy.sleep(0.2)




		

