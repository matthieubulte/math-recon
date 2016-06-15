import Block

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
