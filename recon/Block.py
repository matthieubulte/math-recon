import Rectangle

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
