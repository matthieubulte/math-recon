from shapes import rectangle

class Top:

    def __init__(self, bounding_rectangle):
        self.bounding_rectangle = bounding_rectangle

    def draw(self, target, text=""):
        self.bounding_rectangle.draw(target, text)

    def resolve_fractions(self):
        raise NotImplementedError

    def resolve_exponents(self):
        raise NotImplementedError

    def resolve_indices(self):
        raise NotImplementedError

    def is_exponent(self, other):
        raise NotImplementedError

    def is_index(self, other):
        raise NotImplementedError

    def to_latex(self):
        raise NotImplementedError

    def __str__(self):
        return str(self.bounding_rectangle)

    def __repr__(self):
        return self.__str__()

def find_bounding_box(tops):
    return rectangle.find_bounding_box([ top.bounding_rectangle for top in tops ])
