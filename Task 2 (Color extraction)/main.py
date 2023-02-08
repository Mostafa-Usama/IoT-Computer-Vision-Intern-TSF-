import cv2 as cv
import numpy as np

def colorBar(height, width, color):
    bar = np.zeros((height, width ,3), np.uint8)
    bar[:] = color
    return bar


img = cv.imread("mountain.jpg")
height, width, _ =  img.shape

data = np.reshape(img, (height * width, 3))
data = np.float32(data)

colorsNumber = 4

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.1)
#stop the algorithm iteration if specified accuracy, epsilon, is reached (0.1)
# OR stop the algorithm after the specified number of iterations, max_iter. (10)
flags = cv.KMEANS_RANDOM_CENTERS

_, _, colors = cv.kmeans(data, colorsNumber, None, criteria, 10, flags)

bars = []

for color in colors:
    bar = colorBar(100,100,color)
    bars.append(bar)

img_bar = np.hstack(bars)

cv.imshow("colors",img_bar)
cv.imshow("image", img)

cv.waitKey(0)

