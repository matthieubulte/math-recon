%matplotlib inline

import matplotlib.pyplot as plt

from shapes.image import *
from shapes.rectangle import *
from pieces.block import *
from pieces.fraction import *
from pieces.symbol import *

plt.rcParams['figure.figsize'] = (15, 5)

def disp(image, rectangle):
    cp = image.image.copy()
    rectangle.draw(cp)
    plt.imshow(cp)

image = Image.from_file('images/fraction_3.png')
rectangles = prepare_rectangles(image.find_rectangles())

block = Block([ Symbol(rectangle) for rectangle in rectangles])

block.resolve_fractions()
block.resolve_exponents()
block.resolve_indices()
disp(image, block)

print block.to_latex()
