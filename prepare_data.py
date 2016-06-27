from shapes.image import *
import cv2

image_per_input = 6
filename = "img%03d-%03d.png"
source = "../dataset/"
target = "dataset_2/"

def find_largest_rect(img):
    rs = img.find_rectangles()
    rs.sort(key=lambda r: -r.area())
    return rs[0]

for i in xrange(1, 11):
    for j in xrange(1, 56):
        f = filename % (i, j)
        img = Image.from_file(source + f)

        r = find_largest_rect(img)
        img = img.sub_image(r).center().resize(28, 28, interpolation=cv2.INTER_AREA)

        imgs = img.generate_training_items()
        for (index, img) in enumerate(imgs):
            img.write_to(target + filename % (i, (j - 1) * image_per_input + index + 1))
