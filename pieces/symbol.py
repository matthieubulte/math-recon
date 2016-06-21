from top import Top

class Symbol(Top):
    def __init__(self, bounding_rectangle):
        Top.__init__(self, bounding_rectangle)

    def resolve_fractions(self):
        return

    def resolve_exponents(self):
        return

    def resolve_indices(self):
        return

    def is_exponent(self, other):
        (_, top_right_1, _, _) = self.bounding_rectangle.corners()
        (_, _, _, bottom_left_2) = other.bounding_rectangle.corners()

        center = self.bounding_rectangle.center()
        radius = (self.bounding_rectangle.w + self.bounding_rectangle.h) / 4

        distance = min(top_right_1.distance_to(bottom_left_2), center.distance_to(bottom_left_2))

        if distance < radius and bottom_left_2.y < center.y:
            return True, distance

        return False, 0

    def is_index(self, other):
        (_, _, bottom_right_1, _) = self.bounding_rectangle.corners()
        (top_left_2, _, _, _) = other.bounding_rectangle.corners()

        center = self.bounding_rectangle.center()
        radius = (self.bounding_rectangle.w + self.bounding_rectangle.h) / 4

        distance = min(bottom_right_1.distance_to(top_left_2), center.distance_to(top_left_2))

        if distance < radius and top_left_2.y > center.y:
            return True, distance

        return False, 0

    def to_latex(self, image, classifier):
        return str(classifier.classify(image.sub_image(self.bounding_rectangle).resize(28, 28).image))

    def traverse(self, function):
        function(self.bounding_rectangle)
