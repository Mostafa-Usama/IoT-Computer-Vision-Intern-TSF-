import cv2 as cv
import numpy as np

model = "MobileNetSSD_deploy.caffemodel" # pretrained model
proto_txt = "MobileNetSSD_deploy.prototxt.txt" # description of the network architecture
min_conf = 0.4 #min confidence that an object is detected

classes = ['background',
    'aeroplane', 'bicycle', 'bird', 'boat',
     'bottle', 'bus', 'car', 'cat', 'chair',
     'cow', 'diningtable', 'dog', 'horse',
     'motorbike',  'person',  'pottedplant',
     'sheep', 'sofa', 'train', 'tvmonitor'] # detected object classes

np.random.seed(19) # so random gives the same result everytime

colors = np.random.uniform(0,255,size=(len(classes),3)) #random color for every class
net = cv.dnn.readNetFromCaffe(proto_txt,model) # load model 


img = cv.imread("img3.jpg") # read image for detection
height, width,_ = img.shape # get height and weight of image

blob = cv.dnn.blobFromImage(cv.resize(img,(320,320)),0.007843,(320,320),127.5)
#(Binary Large Object) image with size 320, scale factor, size of image,mean

net.setInput(blob) # set the blob to be the input to the model
results = net.forward()# feed the image to the model to detect the objects

for i in range(results.shape[2]): # loop over all detected objects

    conf = results[0,0,i,2] # get the confidence of the detected object

    if conf >= min_conf: # check if it's more than or equal to the min confidence

        detected = int(results[0,0,i,1]) # the class of the detected object

        box = results[0,0,i,3:7] * np.array([width,height,width,height]) 
        # return the top left point and the bottom right point of the detected 
        # objects and multiply them with the actual
        # width and height (because they are normalized after detection)

        topX, topY, bottomX, bottomY = box.astype("int") # return the coordinates 
        label = str(classes[detected]) + ": " +  str(int((conf *100)))+"%" 
        # set the label text of the detected object 

        
        cv.rectangle(img,(topX,topY),(bottomX,bottomY),colors[detected],2)
        # draw a rectangle around the object

        cv.putText(img,label,(topX,topY-5),cv.FONT_HERSHEY_SIMPLEX,0.5,colors[detected],2)
        # put the label above the rectangle 
        
cv.imshow("Image",img) # show image
cv.waitKey(0) # wait for any key press or the (X) button 