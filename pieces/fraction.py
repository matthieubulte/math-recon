from top import Top, find_bounding_box

class Fraction(Top):

    def __init__(self, over, under):
        Top.__init__(self, find_bounding_box(over + under))

        from block import Block

        self.over = Block(over)
        self.under = Block(under)

    def draw(self, target, text=""):
        if text:
            text = text + "."

        self.over.draw(target, text + "o")
        self.under.draw(target, text + "u")

    def resolve_fractions(self):
        self.over.resolve_fractions()
        self.under.resolve_fractions()

    def resolve_exponents(self):
        self.over.resolve_exponents()
        self.under.resolve_exponents()

    def resolve_indices(self):
        self.over.resolve_indices()
        self.under.resolve_indices()

    def is_exponent(self, other):
        return False, 0

    def is_index(self, other):
        return False, 0

    def to_latex(self):
        return "\\frac{" + self.over.to_latex() + "}{" + self.under.to_latex() + "}"

def from_fraction_rectangle(fraction_symbol, tops):
    over = []
    under = []

    for top in tops:
        if top is fraction_symbol:
            continue

        # check if they're vertically overlapping
        if fraction_symbol.bounding_rectangle.right < top.bounding_rectangle.left or fraction_symbol.bounding_rectangle.left > top.bounding_rectangle.right:
            continue

        if top.bounding_rectangle.top < fraction_symbol.bounding_rectangle.top:
            over.append(top)
        else:
            under.append(top)

    return None if not under and not over else Fraction(over, under)
