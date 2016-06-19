from symbol import Symbol
from top import Top

class IndexedSymbol(Top):
    def __init__(self, base, index):
        Top.__init__(self, base.bounding_rectangle)
        self.base = base
        self.index = index

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

        self.index.draw(target, text + "i")

    def to_latex(self):
        return self.base.to_latex() + "_{" + self.index.to_latex() + "}"
