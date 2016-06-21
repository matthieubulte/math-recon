#%matplotlib inline

import tensorflow as tf

from shapes.image import *
from shapes.rectangle import *
from pieces.block import *
from pieces.fraction import *
from pieces.symbol import *
from classifier import *

import sys

path = None
for (index, value) in enumerate(sys.argv):
    if value == "--image":
        if index + 1 < len(sys.argv):
            path = sys.argv[index + 1]
        else:
            print "Image path not provided"
            exit()

if path is None:
    path = "images/fraction_3.png"


plt.rcParams["figure.figsize"] = (15, 5)

image = Image.from_file(path)

rectangles = prepare_rectangles(image.find_rectangles())

block = Block([ Symbol(rectangle) for rectangle in rectangles])

block.resolve_fractions()
block.resolve_exponents()
block.resolve_indices()


session = tf.InteractiveSession()

classifier = Classifier(session)
classifier.restore_model_from("model.ckpt")


equation = block.to_latex(image, classifier)

template = """
\documentclass{article}

\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{graphicx}

\\begin{document}
\\begin{figure}[ht!]
\centering
\includegraphics[width=90mm] {%s}
\end{figure}
\\begin{align*}
%s
\end{align*}
\end{document}
"""

print template % (path, equation)
