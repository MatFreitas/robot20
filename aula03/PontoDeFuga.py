#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math


# Definindo lista a ser utilizada para os pontos.
lista_pontos = []

# Função que calcula ponto de intersecção.  
def interseccao(x1,y1,x2,y2,x3,y3,x4,y4):
    x1 = x1
    y1 = y1
    x2 = x2
    y2 = y2
    x3 = x3
    y3 = y3
    x4 = x4
    y4 = y4
   
    # definindo coef angulares
    if x2-x1 == 0 or x4-x3 == 0:
        return 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 
    else:
        m1 = (y2-y1)/(x2-x1)
        m2 = (y4-y3)/(x4-x3)
        # definindo coef lineares
        h1 = y1 - m1*x1
        h2 = y3 - m2*x3
        # achando as coordenadas da intersecção
        xi = (h2-h1)/(m1-m2)
        yi = m1*xi+h1
        # ponto de intersecção (xi,yi)
        return xi.round(4), yi.round(4), m1.round(4), m2.round(4), h1.round(4), h2.round(4)

# Vídeos exemplos a serem testados

cap = cv2.VideoCapture('VIDEO1.mp4')
#cap = cv2.VideoCapture('VIDEO2.mp4')
#cap = cv2.VideoCapture('VIDEO3.mp4')


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")

    # Our operations on the frame come here
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor (frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Fazendo a máscara.
    cor1_v2 = np.array([ 225, 225, 225], dtype=np.uint8)
    cor2_v2 = np.array([ 255, 255, 255], dtype=np.uint8)
    mascara = cv2.inRange(rgb, cor1_v2, cor2_v2)
    
    # Usando Canny para detectar todas as linhas da máscara
    min_contrast = 100
    max_contrast = 200
    linhas = cv2.Canny(gray, min_contrast, max_contrast )
    
    hough_img = linhas.copy() # Vamos reusar a imagem de canny


    lines = cv2.HoughLinesP(hough_img, 10, math.pi/180.0, 100, np.array([]), 45, 2.5)
    
    a,b,c = lines.shape
    
    hough_img_rgb = cv2.cvtColor(hough_img, cv2.COLOR_GRAY2BGR)
    
    # Parâmetros que fazem com que adiconem pontos à lista_pontos apenas 
    # duas vezes por frame.
    line1 = 0
    line2 = 0
    
    for i in range(a):
        
        # Pega as coordenadas iniciais e finais do ponto encontrado e calcula
        # seu coeficiente angular.
        coordenadas = lines[i][0]
        x1,y1,x2,y2 = coordenadas
        m = (y2-y1)/(x2-x1)
        
        # CONDIÇÕES PARA VÍDEOS 1 E 2!
        if  0.4 <= m <= 2.7 and line1 == 0:
            
        # CONDIÇÕES PARA VÍDEO 3!
        #if  0.1 <= m <= 0.31 and line1 == 0: 
            
            
            x1 = lines[i][0][0]
            x2 = lines[i][0][2]
            y1 = lines[i][0][1]
            y2 = lines[i][0][3]
            line1 +=1
            
            lista_pontos.append([x1,y1,x2,y2])
        
        # CONDIÇÕES PARA VÍDEOS 1 E 2!
        elif -0.4 >= m >= -2.7 and line2 == 0:
            
        # CONDIÇÕES PARA VÍDEO 3!   
        #elif -0.1 >= m >= -0.31 and line2 ==0: 
            
            
            
            x3 = lines[i][0][0]
            x4 = lines[i][0][2]
            y3 = lines[i][0][1]
            y4 = lines[i][0][3]
            
            line2 +=1
            lista_pontos.append([x3,y3,x4,y4])
            
        # Só irá desenhar quando houver os pontos de duas retas em lista_pontos.    
        if  len(lista_pontos) == 2:
            
            # Calcula o ponto de intersecção das duas retas e pega outros 
            # valores para conseguirmos desenhar as retas na tela.
            x, y, m1, m2, h1, h2 = interseccao(lista_pontos[0][0], lista_pontos[0][1], lista_pontos[0][2], lista_pontos[0][3], lista_pontos[1][0], lista_pontos[1][1], lista_pontos[1][2], lista_pontos[1][3])
            
            # Previne erros de floats infinitas.
            if x == -float('inf') or x == float('inf'):
                x = 0
                y = 0
            
            # Previne erros de NaN floats.
            elif math.isnan(x) == True:
                x = 0
                y = 0
            
            # Transforma em inteiros para que cv2.line possa receber como
            # parâmetro.
            else:
                x = int(x)
                y = int(y)
            
            # Desenha um círculo cujo centro são as coordenadas do ponto
            # de intersecção.
            cv2.circle(frame, (x,y), 10, (255, 0, 0), -1 )
            
            # Desenha as duas retas que vão se inteseccionar.
            cv2.line(frame, (0, int(h1)), (10000, int(m1*10000 + h1)), (0,0,255), 2, cv2.LINE_AA)
            cv2.line(frame, (0, int(h2)), (10000, int(m2*10000 + h2)), (0,0,255), 2, cv2.LINE_AA)
            
            # Esvazia lista_pontos para o próximo loop.
            lista_pontos = []
            
        
            
    cv2.imshow('Ponto de fuga', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
