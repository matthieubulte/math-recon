
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

    def is_horizontal_line(self, fraction_ratio_threshold=0.25):
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

    @staticmethod
    def horizontal_sort(rectangles):
        rectangles.sort(key=lambda rectangle: rectangle.x)

    @staticmethod
    def remove_tiny(rectangles, area_percentage_threshold=0.025):
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
