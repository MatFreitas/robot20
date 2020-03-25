#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Matheus Dib, Fabio de Miranda"


import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from math import atan, degrees
import math
import auxiliar as aux

# If you want to open a video, just change v2.VideoCapture(0) from 0 to the filename, just like below
#cap = cv2.VideoCapture('hall_box_battery.mp4')

# Parameters to use when opening the webcam.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1

print("Press q to QUIT")

# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def center_of_contour(contorno):
    """ Retorna uma tupla (cx, cy) que desenha o centro do contorno"""
    M = cv2.moments(contorno)
    # Usando a expressão do centróide definida em: https://en.wikipedia.org/wiki/Image_moment
    if M["m00"] == 0:
        return (250,250)
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (int(cX), int(cY))

def crosshair(img, point, size, color):
    """ Desenha um crosshair centrado no point.
        point deve ser uma tupla (x,y)
        color é uma tupla R,G,B uint8
    """
    x,y = point
    cv2.line(img,(x - size,y),(x + size,y),color,5)
    cv2.line(img,(x,y - size),(x, y + size),color,5)
    
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #blur = gray
    # Detect the edges present in the image
    bordas = auto_canny(blur)
    
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    circles = []


    # Obtains a version of the edges image where we can draw in color
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)

    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    circles = None
    circles=cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=60)
    
    hsv1, hsv2 = aux.ranges('#ff00ff')
    #hsv1=140 e hsv2=150 após serenm calculados
    hsv1= (120, 50, 50)
    hsv2= (180, 255, 255)
    mascara_magenta = cv2.inRange(img_hsv, hsv1, hsv2)
            
    cor1, cor2 = aux.ranges('#00ffff')
    #cor1=80 e cor2=90 após sererm calculados
    cor1= (60, 50, 50)
    cor2= (110, 255, 255)
    mascara_ciano = cv2.inRange(img_hsv, cor1, cor2)

    if circles is not None:        
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            print(i)
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(bordas_color,(i[0],i[1]),2,(0,0,255),3)
            
    
    # Draw a diagonal blue line with thickness of 5 px
    # cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    cv2.line(bordas_color,(0,0),(511,511),(255,0,0),5)

    # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    cv2.rectangle(bordas_color,(384,0),(510,128),(0,255,0),3)

    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bordas_color,'Press q to quit',(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)

    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
   
    mascara_total = mascara_magenta + mascara_ciano
    
    retorno, mascara_limiar = cv2.threshold(mascara_total, 100 ,255, cv2.THRESH_BINARY)
    
    contornos, arvore = cv2.findContours(mascara_limiar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    mask_rgb = cv2.cvtColor(mascara_limiar, cv2.COLOR_GRAY2RGB) 
    contornos_img = mask_rgb.copy() # Cópia da máscara para ser desenhada "por cima"
    
    lista_cm = []
    for c in contornos:
        p = center_of_contour(c) # centro de massa
        a = cv2.contourArea(c) # área
        if a >500:
            lista_cm.append(p)
    
    if len(lista_cm) == 2:
        cv2.line(contornos_img, lista_cm[0], lista_cm[1], (255, 0, 0), 5)
        
        
 
    
    circles=cv2.HoughCircles(image=bordas,method=cv2.HOUGH_GRADIENT,dp=2.5,minDist=40,param1=50,param2=100,minRadius=5,maxRadius=100)
    mask_limiar_rgb = cv2.cvtColor(mascara_limiar, cv2.COLOR_GRAY2RGB)
    bordas_rgb = cv2.cvtColor(bordas, cv2.COLOR_GRAY2RGB)
    
    output =  contornos_img
    
    if circles is not None:        
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(output,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(output,(i[0],i[1]),2,(0,0,255),3)
                    
     
    
    if lista_cm != [] and len(lista_cm) != 1 and (lista_cm[1][0] - lista_cm[0][0]) != 0:
        angulo= degrees(atan((-1)*(lista_cm[1][1] - lista_cm[0][1])/(lista_cm[1][0] - lista_cm[0][0])))
        cv2.putText(contornos_img,'angulo: {:05f}'.format(angulo),(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
        distanciaPixel = math.sqrt((lista_cm[1][1] - lista_cm[0][1])**2 + (lista_cm[1][0] - lista_cm[0][0])**2)
        distancia = (14*300)/distanciaPixel
        cv2.putText(contornos_img,'distancia: {:05f}'.format(distancia),(0,100), font, 1,(255,255,255),2,cv2.LINE_AA)

    
    # Display the resulting frame
    cv2.imshow('Detector de circulos', contornos_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
