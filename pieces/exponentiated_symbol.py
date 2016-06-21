from symbol import Symbol
from top import Top

class ExponentiatedSymbol(Top):
    def __init__(self, base, exponent):
        Top.__init__(self, base.bounding_rectangle)
        self.base = base
        self.exponent = exponent

    def resolve_fractions(self):
        return

    def resolve_exponents(self):
        return

    def resolve_indices(self):
        return

    def is_exponent(self, other):
        return self.base.is_exponent(other)

    def is_index(self, other):
        return self.base.is_index(other)

    def draw(self, target, text=""):
        self.base.draw(target, text)

        if text:
            text = text + "."

        self.exponent.draw(target, text + "e")

    def to_latex(self, image, classifier):
        return self.base.to_latex(image, classifier) + "^{" + self.exponent.to_latex(image, classifier) + "}"

    def traverse(self, function):
        self.base.traverse(function)
        self.exponent.traverse(function)
