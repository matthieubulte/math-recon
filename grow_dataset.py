%matplotlib inline

import cv2
import numpy as np
from shapes.image import *
from random import random

f = "dataset/img%03d-%03d.png" % (1, 1)

img = Image.from_file(f).image
dst = img

def maybe_flip(rate, x):
    if random() < rate:
        if random() < 0.5:
            return x + 1
        else:
            return x - 1
    else:
        return x

map_x = np.zeros(img.shape[:2],np.float32)
map_y = np.zeros(img.shape[:2],np.float32)
rows,cols = img.shape[:2]

rate = 0.05

for y in xrange(rows):
    for x in xrange(cols):

        nx = maybe_flip(rate, x)
        ny = maybe_flip(rate, y)

        if nx < cols and nx >= 0:
            map_x.itemset((y, x), nx)
            map_x.itemset((y, nx), x)
        if ny < rows and ny >= 0:
            map_y.itemset((y, x), ny)
            map_y.itemset((ny, x), y)



dst = cv2.remap(dst, map_x, map_y, cv2.INTER_LINEAR)
Image(dst).show()
