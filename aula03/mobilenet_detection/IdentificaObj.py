# Para RODAR
# python IdentificaObj.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# Credits: https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

print("Para executar:\npython object_detection_webcam.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel")

# import the necessary packages
import numpy as np
import argparse
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)



def detect(frame, results_category):
    image = frame.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    results = []
    
    
    
    
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        
        #indicador a ser usado no while que diz
        #se reconheceu a classe 'person' ou não 
        index = 0

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        
        if confidence > args["confidence"] and  confidence > 0.9:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[0], confidence * 100)
            print("[INFO] {}".format(label))
            
            #checa se todos a cãmera reconheceu a classe 'person' por 5 frames
            #seguidos
            k = ''
            if len(results_category) >= 5:
                for i in results_category[-5:]:  
                    k+= i
                    
            #desenha o retângulo fixo em volta da pessoa caso 
            #o evento descrito acima tenha sido confirmado.
            if k == 'person'*5:
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    COLORS[0], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)

            results.append((CLASSES[0], confidence*100, (startX, startY),(endX, endY) ))
            
            
        else:
            #Não reconheceu a classe 'person'
            index = 1
       
                
                       
        
         

    # show the output image
    return image, results, index







#cap = cv2.VideoCapture('hall_box_battery_1024.mp4')
cap = cv2.VideoCapture(0)

print("Known classes")
print(CLASSES)


results_category = []

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    result_frame, result_tuples, index = detect(frame, results_category)
    
    
    if result_tuples != []:
        results_category.append(result_tuples[0][0])
   
    #se não reconheceu a classe, torna results_category uma lista vazia 
    #(para 'recomeçar' o processo).
    if index  == 1:
        results_category = []
   
         
    
            
    
    # Display the resulting frame
    cv2.imshow('frame',result_frame)

    # Prints the structures results:
    # Format:
    # ("CLASS", confidence, (x1, y1, x2, y3))
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
