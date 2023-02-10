import cv2 as cv
import numpy as np

def colorBar(height, width, color): #create an image with width, height and color
    bar = np.zeros((height, width ,3), np.uint8)
    bar[:] = color
    return bar


img = cv.imread("mountain.jpg") # read image
height, width, _ =  img.shape # get image's height and width

data = np.reshape(img, (height * width, 3)) #reshape the image array to 2 dimensions(total pixels, colors)
data = np.float32(data) # cast the array to float 

colorsNumber = 4 # n Dominant colors

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1) 
#stop the algorithm iteration if specified accuracy, epsilon, is reached (0.1)
# OR stop the algorithm after the specified number of iterations, max_iter. (10)

flags = cv.KMEANS_RANDOM_CENTERS # initilazie random centers for the Kmean algorithm

_, _, centroid = cv.kmeans(data, colorsNumber, None, criteria, 10, flags) # run kmeans and return the centroid 

bars = [] # store dominant colors

for color in centroid:
    bar = colorBar(100,100,color) # create an image with width 100 height 100 and color
    bars.append(bar) 

img_bar = np.hstack(bars) #create the color pallete

cv.imshow("colors",img_bar) # show color pallete
cv.imshow("image", img) # show original image

cv.waitKey(0)

