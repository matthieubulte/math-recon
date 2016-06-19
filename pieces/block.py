from top import Top, find_bounding_box
from fraction import from_fraction_rectangle
from exponentiated_symbol import ExponentiatedSymbol
from indexed_symbol import IndexedSymbol
from shapes import rectangle
import math

class Block(Top):
    def __init__(self, children):
        Top.__init__(self, find_bounding_box(children))
        self.children = children

    def draw(self, target, text=""):
        if text:
            text = text + "."

        for (index, top) in enumerate(self.children):
            top.draw(target, text + str(index))

    def resolve_fractions(self):
        line_likes = [top for top in self.children if top.bounding_rectangle.is_horizontal_line()]
        line_likes.sort(key=lambda top: top.bounding_rectangle.x)

        if not line_likes:
            return

        for top in line_likes:
            fraction = from_fraction_rectangle(top, self.children)

            if fraction:
                self.children[ self.children.index(top) ] = fraction
                self.__remove_all(fraction.over.children + fraction.under.children)

                fraction.resolve_fractions()
                self.resolve_fractions()
                return

    def resolve_exponents(self):
        self.__resolve_exponents()

        for child in self.children:
            child.resolve_exponents()

    def resolve_indices(self):
        self.__resolve_indices()

        for child in self.children:
            child.resolve_indices()


    def is_exponent(self, other):
        return False, 0

    def is_index(self, other):
        return False, 0

    def to_latex(self):
        return " ".join([top.to_latex() for top in self.children])

    @staticmethod
    def __exponent_predicate(center, other_rectangle):
        return center.y > other_rectangle.bottom

    @staticmethod
    def __index_predicate(center, other_rectangle):
        return other_rectangle.top > center.y

    def __resolve_exponents(self):
        exponent_infos = self.__find_decorated( lambda top, other: top.is_exponent(other) )
        exponent_infos = exponent_infos[::-1]

        # here it seems weird to have "center.y > other_rectangle.bottom" but this is due
        # to having top-left-based coordinates
        exponent_predicate = lambda center, other_rectangle: Block.__exponent_predicate(center, other_rectangle)
        stop_predicate = lambda center, other_rectangle: not Block.__index_predicate(center, other_rectangle)

        for exponent_info in exponent_infos:
            exponent_line = self.__find_line_elements(exponent_info, exponent_predicate, stop_predicate)

            self.children[exponent_info[0]] = ExponentiatedSymbol(self.children[exponent_info[0]], Block(exponent_line))
            self.__remove_all(exponent_line)


    def __resolve_indices(self):
        index_infos = self.__find_decorated( lambda top, other: top.is_index(other) )
        index_infos = index_infos[::-1]

        # here it seems weird to have "center.y < other_rectangle.top" but this is due
        # to having top-left-based coordinates
        index_predicate = lambda center, other_rectangle: Block.__index_predicate(center, other_rectangle)
        stop_predicate = lambda center, other_rectangle: not Block.__exponent_predicate(center, other_rectangle)

        for index_info in index_infos:
            index_line = self.__find_line_elements(index_info, index_predicate, stop_predicate)

            self.children[index_info[0]] = IndexedSymbol(self.children[index_info[0]], Block(index_line))
            self.__remove_all(index_line)

    def __find_line_elements(self, decoration_info, line_predicate, stop_predicate):
        (top_index, decoration_index) = decoration_info

        top_center = self.children[top_index].bounding_rectangle.center()
        decoration_line = [ self.children[decoration_index] ]

        for other in self.children[ decoration_index + 1: ]:
            if line_predicate(top_center, other.bounding_rectangle):
                decoration_line.append(other)
            elif stop_predicate(top_center, other.bounding_rectangle):
                break

        return decoration_line

    def __find_decorated(self, decoration_predicate):
        decorated = []

        for (top_index, top) in enumerate(self.children):
            min_distance = (top.bounding_rectangle.w + top.bounding_rectangle.h) / 4
            closest_decoration = None

            for (other_index, other) in enumerate(self.children[top_index + 1:]):
                is_decoration, distance = decoration_predicate(top, other)

                if is_decoration and distance < min_distance:
                    min_distance = distance
                    closest_decoration = top_index + 1 + other_index

            if closest_decoration is not None:
                decorated.append( (top_index, closest_decoration) )


        return decorated

    def __remove_all(self, symbols):
        self.children = [top for top in self.children if top not in symbols]
