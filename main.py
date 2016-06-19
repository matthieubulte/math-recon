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




# for each rectangles block
# find first power and exponent
def rect_corners(r):
    [x, y, w, h] = r
    return [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]

def rect_center(r):
    [x, y, w, h] = r
    return (int(x + w / 2), int(y + h / 2))

def distance(p1, p2):
    (p1x, p1y) = p1
    (p2x, p2y) = p2

    return (p1x - p2x) ** 2 + (p1y - p2y) ** 2

def find_decorated_symbols(rectangles):
    symbols = []

    for k in range(len(rectangles)):
        (tl_1, tr_1, br_1, bl_1) = rect_corners(rectangles[k])
        [x, y, w, h] = rectangles[k]
        radius = w / 2
        center = rect_center(rectangles[k])

        exp_dist = 100000000
        exp_index = None

        ind_dist = 100000000
        ind_index = None

        for j in range(k + 1, len(rectangles)):
            (tl_2, tr_2, br_2, bl_2) = rect_corners(rectangles[j])

            current_exp_dist = min(distance(tr_1, bl_2), distance(center, bl_2))
            current_ind_dist = min(distance(br_1, tl_2), distance(center, tl_2))

            if current_exp_dist < radius**2 and current_exp_dist < exp_dist and bl_2[1] < center[1]:
                exp_dist = current_exp_dist
                exp_index = j

            if current_ind_dist < radius**2 and current_ind_dist < ind_dist and tl_2[1] > center[1]:
                ind_dist = current_ind_dist
                ind_index = j

        symbols.append((k, exp_index, ind_index))

    return symbols

# find power lines and exponents
# [e] ^ [x] _ [y]
def find_decoration_lines(rectangles, rect_info):
    (index, exp, ind) = rect_info

    rect = rectangles[index]
    (_, center_y) = rect_center(rect)

    exp_parts = []
    ind_parts = []

    if exp is None and ind is None:
        return ([], [])

    for next_rect in rectangles[index+1:]:
        [_, y, _, h] = next_rect

        if exp is not None and y + h < center_y:
            exp_parts.append(next_rect)
        elif ind is not None and  y > center_y:
            ind_parts.append(next_rect)
        elif y < center_y:
            break

    return (exp_parts, ind_parts)
