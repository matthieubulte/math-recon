#%matplotlib inline

from shapes.image import *
import cv2

filename = "img%03d-%03d.png"
source = "../dataset/"
target = "dataset_2/"

for i in range(1, 63):
    for j in range(1, 56):
        f = filename % (i, j)
        img = Image.from_file(source + f)
        r = img.find_rectangles()[0]
        img = img.sub_image(r).resize(28, 28, interpolation=cv2.INTER_AREA)
        img.write_to(target + f)
