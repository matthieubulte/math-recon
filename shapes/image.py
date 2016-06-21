import cv2
import numpy as np
import matplotlib.pyplot as plt

from rectangle import *

class Image:
    def __init__(self, image):
        self.image = image

    def __is_empty_rectangle(self, rectangle, image_average, variance_threshold = 200, background_similarity_threshold = 30):
        rectangle_image = self.sub_image(rectangle)

        has_low_variance = rectangle_image.variance() < variance_threshold
        is_background = abs(rectangle_image.average() - image_average) < background_similarity_threshold

        # remove boxes with high variance and the same color as the background
        return has_low_variance and is_background

    def average(self):
        return np.average(self.image)

    def variance(self):
        return np.var(self.image)

    def sub_image(self, rectangle):
        return Image(self.image[rectangle.y : rectangle.y + rectangle.h, rectangle.x : rectangle.x + rectangle.w])

    def find_rectangles(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blured = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ctrs, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        average = self.average()

        rectangles = [Rectangle.from_array(cv2.boundingRect(ctr)) for ctr in ctrs]

        return [rectangle for rectangle in rectangles if not self.__is_empty_rectangle(rectangle, average) ]

    def display(self):
        plt.imshow(self.image)

    def width(self):
        return self.image.shape[1]

    def height(self):
        return self.image.shape[0]

    def sub_image(self, rectangle):
        return Image(self.image[rectangle.top : rectangle.bottom, rectangle.left : rectangle.right])

    def resize(self, width, height):
        return Image(cv2.resize(self.image, (width, height), interpolation=cv2.INTER_AREA))

    def center(self):
        width, height = self.image.shape[:2]
        size = max(width, height)

        average_color = np.average(np.average(self.image, axis=0), axis=0)

        new_image = np.zeros((size, size, 3), np.uint8)
        new_image[:] = (int(average_color[0]), int(average_color[1]), int(average_color[2]))
        new_image[(size - width) / 2:(size + width) / 2, (size - height) / 2:(size + height) / 2] = self.image

        return Image(new_image)


    def write_to(self, path):
        cv2.imwrite(path, self.image)

    def show(self):
        plt.imshow(self.image)

    @staticmethod
    def from_file(image_path):
        return Image(cv2.imread(image_path))
