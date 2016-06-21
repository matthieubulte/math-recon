%matplotlib inline

import tensorflow as tf

from shapes.image import *
from shapes.rectangle import *
from pieces.block import *
from pieces.fraction import *
from pieces.symbol import *
from classifier import *

plt.rcParams['figure.figsize'] = (15, 5)


image = Image.from_file('images/fraction_3.png')

image.show()

rectangles = prepare_rectangles(image.find_rectangles())

block = Block([ Symbol(rectangle) for rectangle in rectangles])

block.resolve_fractions()
block.resolve_exponents()
block.resolve_indices()


session = tf.InteractiveSession()

classifier = Classifier(session)
classifier.restore_model_from("model.ckpt")


print block.to_latex(image, classifier)
