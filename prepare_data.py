from shapes.image import *
import cv2

filename = "img%03d-%03d.png"
source = "../dataset/"
target = "datasets/"

def find_largest_rect(img):
    rs = img.find_rectangles()
    rs.sort(key=lambda r: -r.area())
    return rs[0]

def generate_training_items(image):
    def make_distortions(img):
        return [
            img,
            img.distort(0.1),
            img.distort(0.2)
        ]

    return make_distortions(image.rotate(-20)) + \
        make_distortions(image.rotate(-10)) + \
        make_distortions(image) + \
        make_distortions(image.rotate(10)) + \
        make_distortions(image.rotate(20))


image_per_input = 15
for i in xrange(1, 11):
    for j in xrange(1, 56):
        f = filename % (i, j)
        img = Image.from_file(source + f)

        r = find_largest_rect(img)
        img = img.sub_image(r).center().resize(28, 28, interpolation=cv2.INTER_AREA)

        imgs = generate_training_items(img)

        for (index, img) in enumerate(imgs):
            img.write_to(target + filename % (i, (j - 1) * image_per_input + index + 1))
