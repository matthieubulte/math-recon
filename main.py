%matplotlib inline

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (15, 5)

class Image:
    def __init__(self, image):
        self.image = image

    def __is_empty_rectangle(self, rectangle, image_average):
        global variance_threshold
        global background_similarity_threshold

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

    @staticmethod
    def from_file(image_path):
        return Image(cv2.imread(image_path))

class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.left = x
        self.top  = y
        self.right = x + w
        self.bottom = y + h

    def draw(self, target, color=(0,0,0), text=""):
        cv2.rectangle(target, (self.x, self.y), (self.right, self.bottom), color)
        cv2.putText(target, text, (self.x - 20, self.y), cv2.FONT_HERSHEY_SIMPLEX, 1, color)

    def intersection(self, other):
        left = max(self.left, other.left)
        right = min(self.right, other.right)

        if left > right:
            return None

        top = max(self.top, other.top)
        bottom = min(self.bottom, other.bottom)

        if top > bottom:
            return None

        return Rectangle(left, top, right - left, bottom - top)

    def is_horizontal_line(self):
        global fraction_ratio_threshold
        return self.w * fraction_ratio_threshold > self.h

    def area(self):
        return self.w * self.h

    def remove_all(self, to_remove):
        self.children = [child for child in self.children if child not in to_remove]

    def __eq__(self, other):
        if not isinstance(other, Rectangle):
            return False
        return self.x is other.x and self.y is other.y and self.w is other.w and self.h is other.h

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "{%s, %s, %s, %s}" % (self.x, self.y, self.w, self.h)

    def __repr__(self):
        return str(self)

    @staticmethod
    def merge_overlapping(rectangles):
        global area_overlapping_threshold

        to_remove = []
        for (index_1, rectangle_1) in enumerate(rectangles):
            area_1 = rectangle_1.area()

            for (index_2, rectangle_2) in enumerate(rectangles[index_1 + 1:]):
                intersection = rectangle_1.intersection(rectangle_2)

                if intersection is None:
                    continue

                area_2 = rectangle_2.area()
                min_area = min(area_1, area_2)

                if intersection.area() > area_overlapping_threshold * min_area:
                    if area_1 < area_2:
                        to_remove.append(rectangle_1)
                        continue
                    else:
                        to_remove.append(rectangle_2)

        return [rectangle for rectangle in rectangles if rectangle not in to_remove]

    @staticmethod
    def horizontal_sort(rectangles):
        rectangles.sort(key=lambda rectangle: rectangle.x)

    @staticmethod
    def remove_tiny(rectangles):
        global area_percentage_threshold

        threshold = Rectangle.middle_area(rectangles) * area_percentage_threshold
        return [rectangle for rectangle in rectangles if rectangle.area() > threshold]

    @staticmethod
    def from_array(rectangle):
        return Rectangle(rectangle[0], rectangle[1], rectangle[2], rectangle[3])

    @staticmethod
    def find_bounding_box(rectangles):
        if not rectangles:
            return EmptyRectangle()

        left = min(rectangles, key=lambda rectangle: rectangle.left).left
        right = max(rectangles, key=lambda rectangle: rectangle.right).right
        top = min(rectangles, key=lambda rectangle: rectangle.top).top
        bottom = max(rectangles, key=lambda rectangle: rectangle.bottom).bottom

        return Rectangle(left, top, right - left, bottom - top)

    @staticmethod
    def middle_area(rectangles):
        areas = [rectangle.area() for rectangle in rectangles]
        return (min(areas) + max(areas)) * .5

class EmptyRectangle(Rectangle):
    def __init__(self):
        Rectangle.__init__(self, 0, 0, 0, 0)

    def draw(self, target, color=(0,0,0), text=""):
        return

class Block(Rectangle):
    def __init__(self, children):
        bounding_rectangle = Rectangle.find_bounding_box(children)
        Rectangle.__init__(self, bounding_rectangle.x, bounding_rectangle.y, bounding_rectangle.w, bounding_rectangle.h)

        self.children = children

    def draw(self, target, color=(0,0,0), text=""):
        (r, g, b) = color

        if text:
            text = text + "."

        for (index, child) in enumerate(self.children):
            child.draw(target, color=(r + 25, g + 25, b + 25), text=text + str(index))

    def find_fraction_blocks(self):
        lines = [rectangle for rectangle in self.children if rectangle.is_horizontal_line()]

        # start with the left-most fraction
        lines.sort(key=lambda rectangle: rectangle.x)

        if not lines:
            return

        for line in lines:
            fraction = FractionBlock.from_fraction_rectangle(line, self.children)
            fraction_line = line

            if fraction:
                self.children[ self.children.index(line) ] = fraction
                break

        if not fraction:
            return

        self.remove_all(fraction.children)
        self.find_fraction_blocks()

class FractionBlock(Block):
    def __init__(self, over, under):
        Block.__init__(self, over + under)

        self.over = Block(over)
        self.over.find_fraction_blocks()

        self.under = Block(under)
        self.under.find_fraction_blocks()

    def draw(self, target, color=(0,0,0), text=""):
        (r, g, b) = color

        if text:
            text = text + "."

        self.over.draw(target, color=(r + 25, g + 25, b + 25), text=text + "o")
        self.under.draw(target, color=(r + 25, g + 25, b + 25), text=text + "u")

    @staticmethod
    def from_fraction_rectangle(fraction_rectangle, rectangles):
        over = []
        under = []

        for rectangle in rectangles:
            if rectangle is fraction_rectangle:
                continue

            # check if they're vertically overlapping
            if fraction_rectangle.right < rectangle.left or fraction_rectangle.left > rectangle.right:
                continue

            if rectangle.top < fraction_rectangle.top:
                over.append(rectangle)
            else:
                under.append(rectangle)

        return None if not under and not over else FractionBlock(over, under)

# parameters
area_percentage_threshold = 0.025
variance_threshold = 200
background_similarity_threshold = 30
area_overlapping_threshold = 0.25
fraction_ratio_threshold = 0.25

#
image = Image.from_file('fraction_3.png')

def disp(image, rectangle):
    cp = image.image.copy()
    rectangle.draw(cp)
    plt.imshow(cp)

rectangles = image.find_rectangles()

rectangles = Rectangle.remove_tiny(rectangles)
rectangles = Rectangle.merge_overlapping(rectangles)
Rectangle.horizontal_sort(rectangles)

block = Block(rectangles)
block.find_fraction_blocks()

disp(image, block)

# for each rectangles block
# find first power and exponent
def rect_corners(r):
    [x, y, w, h] = r
    return [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]

def rect_center(r):
    [x, y, w, h] = r
    return (int(x + w / 2), int(y + h / 2))

def distance(p1, p2):
    (p1x, p1y) = p1
    (p2x, p2y) = p2

    return (p1x - p2x) ** 2 + (p1y - p2y) ** 2

def find_decorated_symbols(rectangles):
    symbols = []

    for k in range(len(rectangles)):
        (tl_1, tr_1, br_1, bl_1) = rect_corners(rectangles[k])
        [x, y, w, h] = rectangles[k]
        radius = w / 2
        center = rect_center(rectangles[k])

        exp_dist = 100000000
        exp_index = None

        ind_dist = 100000000
        ind_index = None

        for j in range(k + 1, len(rectangles)):
            (tl_2, tr_2, br_2, bl_2) = rect_corners(rectangles[j])

            current_exp_dist = min(distance(tr_1, bl_2), distance(center, bl_2))
            current_ind_dist = min(distance(br_1, tl_2), distance(center, tl_2))

            if current_exp_dist < radius**2 and current_exp_dist < exp_dist and bl_2[1] < center[1]:
                exp_dist = current_exp_dist
                exp_index = j

            if current_ind_dist < radius**2 and current_ind_dist < ind_dist and tl_2[1] > center[1]:
                ind_dist = current_ind_dist
                ind_index = j

        symbols.append((k, exp_index, ind_index))

    return symbols

# find power lines and exponents
# [e] ^ [x] _ [y]
def find_decoration_lines(rectangles, rect_info):
    (index, exp, ind) = rect_info

    rect = rectangles[index]
    (_, center_y) = rect_center(rect)

    exp_parts = []
    ind_parts = []

    if exp is None and ind is None:
        return ([], [])

    for next_rect in rectangles[index+1:]:
        [_, y, _, h] = next_rect

        if exp is not None and y + h < center_y:
            exp_parts.append(next_rect)
        elif ind is not None and  y > center_y:
            ind_parts.append(next_rect)
        elif y < center_y:
            break

    return (exp_parts, ind_parts)
