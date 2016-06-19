from point import *
import cv2

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

    def draw(self, target, text=""):
        cv2.rectangle(target, (self.x, self.y), (self.right, self.bottom), (0,0,0))
        cv2.putText(target, text, (self.x, self.y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))

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

    def is_horizontal_line(self, fraction_ratio_threshold=0.25):
        return self.w * fraction_ratio_threshold > self.h

    def area(self):
        return self.w * self.h

    def bottom_left(self):
        return Point(self.left, self.bottom)

    def top_left(self):
        return Point(self.left, self.top)

    def corners(self):
        return ( self.top_left(), Point(self.right, self.top), Point(self.right, self.bottom), self.bottom_left() )

    def center(self):
        return Point((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def overlap(self, other, area_overlapping_threshold=0.25):
        intersection = self.intersection(other)

        my_area = self.area()
        other_area = other.area()
        min_area = min(my_area, other_area)

        if intersection.area() > area_overlapping_threshold * min_area:
            if my_area < other_area:
                return True, self
            else:
                return True, other

        return False, None

    def __eq__(self, other):
        if not isinstance(other, Rectangle):
            return False
        return self.x is other.x and self.y is other.y and self.w is other.w and self.h is other.h

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "{%s, %s, %s, %s}" % (self.x, self.y, self.w, self.h)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_array(rectangle):
        return Rectangle(rectangle[0], rectangle[1], rectangle[2], rectangle[3])

class EmptyRectangle(Rectangle):
    def __init__(self):
        Rectangle.__init__(self, 0, 0, 0, 0)

    def draw(self, target, text=""):
        return

def merge_overlapping(rectangles, area_overlapping_threshold=0.25):
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

def horizontal_sort(rectangles):
    rectangles.sort(key=lambda rectangle: rectangle.x)

def remove_tiny(rectangles, area_percentage_threshold=0.025):
    threshold = middle_area(rectangles) * area_percentage_threshold
    return [rectangle for rectangle in rectangles if rectangle.area() > threshold]

def find_bounding_box(rectangles):
    if not rectangles:
        return EmptyRectangle()

    left = min(rectangles, key=lambda rectangle: rectangle.left).left
    right = max(rectangles, key=lambda rectangle: rectangle.right).right
    top = min(rectangles, key=lambda rectangle: rectangle.top).top
    bottom = max(rectangles, key=lambda rectangle: rectangle.bottom).bottom

    return Rectangle(left, top, right - left, bottom - top)

def middle_area(rectangles):
    areas = [rectangle.area() for rectangle in rectangles]
    return (min(areas) + max(areas)) * .5

def prepare_rectangles(rectangles):
    rectangles = remove_tiny(rectangles)
    rectangles = merge_overlapping(rectangles)
    horizontal_sort(rectangles)
    return rectangles
