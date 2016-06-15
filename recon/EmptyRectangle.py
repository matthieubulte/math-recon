import Rectangle

class EmptyRectangle(Rectangle):
    def __init__(self):
        Rectangle.__init__(self, 0, 0, 0, 0)

    def draw(self, target, color=(0,0,0), text=""):
        return
